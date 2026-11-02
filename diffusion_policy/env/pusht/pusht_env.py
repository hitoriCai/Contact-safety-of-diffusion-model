import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


# 将物理引擎 Pymunk 中的物体形状转换成几何计算库 Shapely 中的对象
# 根据T型积木与目标区域的重叠面积计算reward
def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            safety_stress_obstacles=False,
            # 新增参数：随机障碍物
            num_obstacles=1,  # 障碍物数量
            obstacle_radius_min=20,  # 障碍物最小半径
            obstacle_radius_max=40,  # 障碍物最大半径
            obstacle_color='DarkGray',  # 障碍物颜色
            obstacle_mass=1  # 障碍物质量
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        self.collision = False
        # 存储障碍物参数
        self.num_obstacles = num_obstacles
        self.obstacle_radius_min = obstacle_radius_min
        self.obstacle_radius_max = obstacle_radius_max
        self.obstacle_color = obstacle_color
        self.obstacle_mass = obstacle_mass

        # agent_pos, block_pos, block_angle
        # agent机器人每一时刻可以看到的数据范围和格式
        # 5维的向量：[agent_x, agent_y, block_x, block_y, block_angle]
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        # Agent 想要移动去的目标位置坐标 [target_x, target_y]
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog  # 物体重心
        # self.block.center = None
        self.damping = damping  # 物体阻尼
        self.render_action = render_action  # True时，标记一个红叉marker，表示agent尝试前往的目标点

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.safety_stress_obstacles = bool(safety_stress_obstacles)
        # Optional explicit benchmark obstacle. Keeping this separate from the
        # legacy deterministic corridor generator preserves old experiments.
        self.safety_stress_obstacle_spec = None
        # Optional per-scenario goal. None preserves the original fixed goal.
        self.safety_stress_goal_pose = None
    
    def reset(self):
        seed = self._seed
        self._setup()
        if self.safety_stress_goal_pose is not None:
            goal_pose = np.asarray(
                self.safety_stress_goal_pose, dtype=np.float64)
            if goal_pose.shape != (3,) or not np.all(np.isfinite(goal_pose)):
                raise ValueError(
                    f'Invalid explicit safety-stress goal pose: '
                    f'{self.safety_stress_goal_pose!r}')
            self.goal_pose = goal_pose.copy()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)
        # 【关键修复 1】障碍物必须在 _set_state 之后再放置；
        # 否则 _setup() 用的是 add_tee 时硬编码的 (256, 300)，
        # 和真正随机化后的 block 位置完全无关，会导致初始 T 与 obstacle 重叠。
        if self.safety_stress_obstacles:
            self._place_safety_stress_obstacles()
        else:
            self._place_obstacles()

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        self.check_collision()
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step,
            'obstacle_positions': np.array([obstacle.body.position for obstacle in self.obstacles]),
            'obstacle_radius': np.array([obstacle.radius for obstacle in self.obstacles]),
            'is_collision': self.collision,
            }
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)
        
        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        
        # 新增：计算 T 形物体的几何中心 (Bounding Box Center) 
        # 1. 获取所有形状的本地顶点
        all_verts = []
        for shape in self.block.shapes:
            if isinstance(shape, pymunk.Poly):
                all_verts.extend(shape.get_vertices())
        # 2. 计算本地坐标系下的 min/max
        ax = [v.x for v in all_verts]
        ay = [v.y for v in all_verts]
        local_center = Vec2d((min(ax) + max(ax)) / 2, (min(ay) + max(ay)) / 2)
        self.block.center = local_center
        
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # 添加随机圆形障碍物（实际放置移动到 reset() 中 _set_state 之后调用 _place_obstacles）
        self.obstacles = []

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _place_obstacles(self):
        """根据真实 block / agent / goal 位置生成不会重叠的圆形障碍物。

        相对 _setup 中原版的两点修正：
          1) 现在被 reset() 在 _set_state(state) 之后调用，block_pos 是真实随机化后的位置；
          2) 用 T 的几何中心 (block.local_to_world(self.block.center)) 作为距离基准，
             而不是 body 锚点 self.block.position（后者并不在 T 的中心）。
        """
        rs = np.random.RandomState(seed=self._seed)
        margin = self.obstacle_radius_max + 10  # 边缘留出的空间
        min_distance = 60                       # 障碍物 vs agent 的最小距离
        # T 外接圆半径约 78，再加 obstacle 最大半径与一点裕量，保证生成时绝不重叠
        min_tee_distance = 78 + self.obstacle_radius_max + 15

        # 真实位置（注意：block.position 是 body 锚点，不是 T 几何中心）
        agent_pos = np.array(self.agent.position)
        block_center = np.array(self.block.local_to_world(self.block.center))
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_local_center = self.block.center
        goal_center = np.array(goal_body.local_to_world(goal_local_center))

        for _ in range(self.num_obstacles):
            radius = rs.uniform(self.obstacle_radius_min, self.obstacle_radius_max)

            # 尝试生成不与其他物体重叠的位置
            max_attempts = 200
            placed = False
            for _ in range(max_attempts):
                x = rs.uniform(margin, self.window_size - margin)
                y = rs.uniform(margin, self.window_size - margin)
                pos = np.array([x, y])

                if (np.linalg.norm(pos - agent_pos) > min_distance and
                        np.linalg.norm(pos - block_center) > min_tee_distance and
                        np.linalg.norm(pos - goal_center) > min_tee_distance):
                    if len(self.obstacles) == 0:
                        placed = True
                        break

                    collision = False
                    for obstacle in self.obstacles:
                        obs_pos = np.array(obstacle.body.position)
                        obs_radius = obstacle.radius
                        if np.linalg.norm(pos - obs_pos) < (radius + obs_radius + 70):
                            collision = True
                            break

                    if not collision:
                        placed = True
                        break

            if not placed:
                print('obstacle not placed!')
                # 兜底：实在采样不出来就跳过这个障碍物，避免硬塞到 T 上
                continue

            obstacle = self.add_static_circle((x, y), radius, self.obstacle_color)
            self.obstacles.append(obstacle)

    def _place_safety_stress_obstacles(self):
        """Place one deterministic obstacle directly in the task corridor.

        This mode is opt-in and intended only for paired safety evaluation.
        The obstacle lies near the middle of the geometric-center segment from
        the initial T pose to the fixed goal. Selected benchmark seeds ensure
        that both endpoints are collision-free and that a bypass exists.
        """
        if self.safety_stress_obstacle_spec is not None:
            spec = self.safety_stress_obstacle_spec
            position = np.asarray(spec['position'], dtype=np.float64)
            radius = float(spec['radius'])
            if position.shape != (2,) or radius <= 0.0:
                raise ValueError(
                    f'Invalid explicit safety-stress obstacle: {spec!r}')
            if (np.any(position < radius + 10.0)
                    or np.any(position > self.window_size - radius - 10.0)):
                raise RuntimeError(
                    f'Explicit safety-stress obstacle is outside the workspace: '
                    f'{position}.')
            obstacle = self.add_static_circle(
                tuple(position), radius, self.obstacle_color)
            self.obstacles.append(obstacle)
            return

        rng = np.random.RandomState(seed=int(self._seed) ^ 0x5A17)
        radius = float(rng.uniform(20.0, 25.0))
        segment_t = float(rng.uniform(0.49, 0.51))
        signed_offset = float(rng.uniform(-6.0, 6.0))

        start = np.asarray(
            self.block.local_to_world(self.block.center), dtype=np.float64)
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal = np.asarray(
            goal_body.local_to_world(self.block.center), dtype=np.float64)
        delta = goal - start
        length = float(np.linalg.norm(delta))
        if length <= 0.0:
            raise RuntimeError('Safety-stress start and goal centers coincide.')
        unit = delta / length
        normal = np.array([-unit[1], unit[0]], dtype=np.float64)
        position = start + segment_t * delta + signed_offset * normal
        if (np.any(position < radius + 10.0)
                or np.any(position > self.window_size - radius - 10.0)):
            raise RuntimeError(
                f'Safety-stress obstacle is outside the workspace: {position}.')
        obstacle = self.add_static_circle(
            tuple(position), radius, self.obstacle_color)
        self.obstacles.append(obstacle)

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1

        self.space.add(body, shape1, shape2)
        return body

    def add_static_circle(self, position, radius, color='DarkGray'):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.sensor = True
        shape.color = pygame.Color(color)
        self.space.add(body, shape)
        return shape

    def add_dynamic_circle(self, position, radius, mass, color='DarkGray'):
        # 计算圆形的转动惯量
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        # 创建动态物体（可推动）
        body = pymunk.Body(mass, inertia)
        body.position = position
        # 添加摩擦力以使其行为更真实
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color(color)
        # 弹性系数，控制碰撞后的反弹程度
        shape.elasticity = 0.3
        # 摩擦系数
        shape.friction = 0.8
        self.space.add(body, shape)
        return shape

    def get_obstacle_message(self):
        '''
        获取障碍物的位置和半径信息
        
        Returns:
            np.ndarray: 障碍物的位置信息，形状为 (num_obstacles, 2)
            np.ndarray: 障碍物的半径信息，形状为 (num_obstacles,)
        '''
        return np.array([obstacle.body.position for obstacle in self.obstacles]), np.array([obstacle.radius for obstacle in self.obstacles])
    
    '''
    def check_collision(self):
        if self.collision:
            return
        # agent_pos = np.asarray(self.agent.position)
        block_pos = np.array(self.block.position)
        for obs in self.obstacles:
            obs_pos = np.asarray(obs.body.position)
            if (np.linalg.norm(block_pos-obs_pos)-(obs.radius*1.05)) < 0:
                self.collision = True
                return
    '''
    def check_collision(self):
        if self.collision:
            return

        # 1. 遍历 T 型积木的每一个零件（横梁和竖杆）
        for block_shape in self.block.shapes:
            
            # 2. 调用 Space 的 shape_query 方法
            # 这个方法的意思是：在当前空间中，找出所有与 block_shape 重叠的形状
            # 它返回的是一个 ShapeQueryInfo 对象的列表
            intersections = self.space.shape_query(block_shape)
            
            for info in intersections:
                # info.shape 就是那个被撞到的形状
                # 3. 检查这个被撞到的形状是不是我们定义的障碍物
                if info.shape in self.obstacles:
                    self.collision = True
                    return

    '''新增 contact safety'''
    def get_block_pose_message(self):
        '''
        获取当前 T 型积木的位姿信息（位置和角度）
        
        Returns:
            np.ndarray: 积木中心的位置信息，形状为 (2,) -> [x, y]
            float: 积木的旋转角度（弧度）
        '''
        # self.block 是 Pymunk Body 对象
        position = np.array(self.block.position)
        angle = float(self.block.angle)
        
        # 固定的本地几何中心偏移 [0, 60]
        local_center = Vec2d(0, 60) 
        # 【核心】利用物理引擎当前的 pos 和 angle，将 local_center 转换到世界坐标
        # 这一步会随着 self.block 的移动和旋转自动得出正确结果
        cog = np.array(self.block.local_to_world(local_center))

        return position, angle, cog
