class RoboschoolForwardWalker6(SharedMemoryClientEnv):

    HEADSTILL_INIT = 3

    def __init__(self, power):
        self.power = power
        self.camera_x = 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.camera_x = 0
        self.camera_y = 4.3
        self.camera_z = 45.0
        self.camera_follow = 0

        self.prev_headpos = 0
        self.curr_headpos = 0
        self.headstill = self.HEADSTILL_INIT

        self.timetime = 0

        self.gheadpos1 = self.gen_headpos1()
        self.gheadpos2 = self.gen_headpos2()



    def create_single_player_scene(self):
        return SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)

    def robot_specific_reset(self):
        for j in self.ordered_joints:
            j.reset_current_position(self.np_random.uniform( low=-0.1, high=0.1 ), 0)
        self.feet = [self.parts[f] for f in self.foot_list]
        self.feet_contact = np.array([0.0 for f in self.foot_list], dtype=np.float32)
        self.scene.actor_introduce(self)
        self.initial_z = None

    def move_robot(self, init_x, init_y, init_z):
        "Used by multiplayer stadium to move sideways, to another running lane."
        self.cpp_robot.query_position()
        pose = self.cpp_robot.root_part.pose()
        pose.move_xyz(init_x, init_y, init_z)  # Works because robot loads around (0,0,0), and some robots have z != 0 that is left intact
        self.cpp_robot.set_pose(pose)
        self.start_pos_x, self.start_pos_y, self.start_pos_z = init_x, init_y, init_z

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for n,j in enumerate(self.ordered_joints):
            j.set_motor_torque( self.power*j.power_coef*float(np.clip(a[n], -1, +1)) )

    def calc_state(self):
        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()


        #pdb.set_trace()

        #headposes = np.array([self.prev_headpos, self.parts["head"].pose().xyz()])
        #headposes = np.array([1])#np.array([self.parts["head"].pose().xyz()])
        # even elements [0::2] position, scaled to -1..+1 between limits
        # odd elements  [1::2] angular speed, scaled to show -1..+1
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array( [p.pose().xyz() for p in self.parts.values()] ).flatten()
        self.body_xyz = (parts_xyz[0::3].mean(), parts_xyz[1::3].mean(), body_pose.xyz()[2])  # torso z is more informative than mean z
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        r, p, yaw = self.body_rpy
        if self.initial_z is None:
            self.initial_z = z
        self.walk_target_theta = np.arctan2( self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0] )
        self.walk_target_dist  = np.linalg.norm( [self.walk_target_y - self.body_xyz[1], self.walk_target_x - self.body_xyz[0]] )
        self.angle_to_target = self.walk_target_theta - yaw

        self.rot_minus_yaw = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw),  np.cos(-yaw), 0],
             [           0,             0, 1]]
            )
        vx, vy, vz = np.dot(self.rot_minus_yaw, self.robot_body.speed())  # rotate speed back to body point of view

        more = np.array([
            z-self.initial_z,
            np.sin(self.angle_to_target), np.cos(self.angle_to_target),
            0.3*vx, 0.3*vy, 0.3*vz,    # 0.3 is just scaling typical speed into -1..+1, no physical sense here
            r, p], dtype=np.float32)
        return np.clip( np.concatenate([more] + [j] + [self.feet_contact] ), -5, +5)

    def calc_potential(self):
        # progress in potential field is speed*dt, typical speed is about 2-3 meter per second, this potential will change 2-3 per frame (not per second),
        # all rewards have rew/frame units and close to 1.0
        return - self.walk_target_dist / self.scene.dt

    def calc_headpos_rew(self, head_dist):
        if head_dist < 0.005:
            rew = 1.2
        elif head_dist < 0.01:
            rew = 0.7
        elif head_dist < 0.02:
            rew = 0.4
        elif head_dist < 0.03:
            rew = 0.2
        else:
            rew = 0
        return rew

    def calc_headpos_rew2(self, head_dist):
        if head_dist < 0.05:
            rew = 1.2
        elif head_dist < 0.1:
            rew = 0.7
        elif head_dist < 0.2:
            rew = 0.4
        elif head_dist < 0.3:
            rew = 0.2
        else:
            rew = 0
        return rew

    def gen_headpos1(self):
        x_points = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 1, 10]
        y_points = [2, 1.2, 0.7, 0.4, 0.2, 0, 0, 0, 0, 0]

        tck = interpolate.splrep(x_points, y_points)
        def retfunc(x):
            return interpolate.splev(x, tck)
        return retfunc
    def gen_headpos2(self):
        x_points = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 10, 100]
        y_points = [2, 1.2, 0.7, 0.4, 0.2, 0, 0, 0, 0, 0]

        tck = interpolate.splrep(x_points, y_points)
        def retfunc(x):
            return interpolate.splev(x, tck)
        return retfunc


    electricity_cost     = -0.9    # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
    stall_torque_cost    = -0.1    # cost for running electric current through a motor even at zero rotational speed, small
    foot_collision_cost  = -1.0    # touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
    foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
    joints_at_limit_cost = -0.2    # discourage stuck joints

    headstill_reward = 0           #encourage keeping head still for long periods of time
    headstill_const_1 = 0.3
    headstill_const_2 = 0.1

    def step(self, a):
        self.prev_headpos = self.parts["head"].pose().xyz()
        self.prev_headpos3 = self.parts["head"].pose().rpy()[1]
        self.prev_headspeed = self.parts["head"].speed()

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()  # also calculates self.joints_at_limit
        self.timetime += 1

        self.curr_headpos = self.parts["head"].pose().xyz()
        self.curr_headpos3 = self.parts["head"].pose().rpy()[1]
        self.curr_headspeed = self.parts["head"].speed()

        #ipdb.set_trace()

        #print("Prev_headpos: " , self.prev_headpos , "\tCurrent_headpos: ", self.curr_headpos) #dfgdfgrdfgdggd

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        potential_old = self.potential
        self.potential = self.calc_potential()
        progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            #print("CONTACT OF '%s' WITH %s" % (f.name, ",".join(contact_names)) )
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0
            if contact_names - self.foot_ground_object_names:
                feet_collision_cost += self.foot_collision_cost

        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.joints_at_limit)

        headstill_reward = self.headstill_reward
        np_prev_headpos = np.asarray(self.prev_headpos)
        np_curr_headpos = np.asarray(self.curr_headpos)


        np_prev_headpos3 = np.asarray(self.prev_headpos3)
        np_curr_headpos3 = np.asarray(self.curr_headpos3)

        np_prev_headspeed = np.asarray(self.prev_headspeed)
        np_curr_headspeed = np.asarray(self.curr_headspeed)


        #if self.gheadpos1 == None:
        self.gheadpos1 = self.gen_headpos1()
        self.gheadpos2 = self.gen_headpos2()
        #ipdb.set_trace()s
        headpos_dist = np.linalg.norm(np_prev_headpos[2]-np_curr_headpos[2])

        headpos_dist2 = np.linalg.norm(self.parts["head"].pose().rpy()[1])

        headpos_dist3 = np.linalg.norm(np_prev_headpos3-np_curr_headpos3)


        #print("Headspeed Distance: ", np_curr_headspeed) #- np_prev_headspeed)
        #print("headpos_dist3: ", headpos_dist3)


        #headpos_rew = - self.headstill_const_1 * np.power(headpos_dist , 1/8)
        #if self.timetime % 100 == 0:
        #    ipdb.set_trace()

        #headpos_rew = self.headstill_const_1 * self.calc_headpos_rew(headpos_dist)
        #headpos_rew2 = self.headstill_const_2 * self.calc_headpos_rew2(headpos_dist2)
        headpos_rew = self.headstill_const_1 * self.gheadpos1(headpos_dist)
        #headpos_rew2 = self.headstill_const_2 * self.gheadpose(headpos_dist2)
        headpos_rew2 = self.headstill_const_2 * self.gheadpos2(headpos_dist2)

        #headpos_rew = 0
        #headpos_rew2 = 0

        self.rewards = [
            alive,
            progress,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost,
            headpos_rew,
            headpos_rew2 #headpos_dist*10 # sdfgsgsgsdgsdgsdgdgsdgsdgsdgsdgsdgsdgsdgsdgsaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
            ]

        self.frame  += 1
        if (done and not self.done) or self.frame==self.spec.max_episode_steps:
            self.episode_over(self.frame)
        self.done   += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)
        return state, sum(self.rewards), bool(done), {}



    def episode_over(self, frames):
        pass

    def camera_adjust(self):
        #self.camera_dramatic()
        self.camera_simple_follow()

    def camera_simple_follow(self):
        x, y, z = self.body_xyz
        self.camera_x = 0.98*self.camera_x + (1-0.98)*x
        self.camera.move_and_look_at(self.camera_x, y-2.0, 1.4, x, y, 1.0)

    def camera_dramatic(self):
        pose = self.robot_body.pose()
        speed = self.robot_body.speed()
        x, y, z = pose.xyz()
        if 1:
            camx, camy, camz = speed[0], speed[1], 2.2
        else:
            camx, camy, camz = self.walk_target_x - x, self.walk_target_y - y, 2.2

        n = np.linalg.norm([camx, camy])
        if n > 2.0 and self.frame > 50:
            self.camera_follow = 1
        if n < 0.5:
            self.camera_follow = 0
        if self.camera_follow:
            camx /= 0.1 + n
            camx *= 2.2
            camy /= 0.1 + n
            camy *= 2.8
            if self.frame < 1000:
                camx *= -1
                camy *= -1
            camx += x
            camy += y
            camz  = 1.8
        else:
            camx = x
            camy = y + 4.3
            camz = 2.2
        #print("%05i" % self.frame, self.camera_follow, camy)
        smoothness = 0.97
        self.camera_x = smoothness*self.camera_x + (1-smoothness)*camx
        self.camera_y = smoothness*self.camera_y + (1-smoothness)*camy
        self.camera_z = smoothness*self.camera_z + (1-smoothness)*camz
        self.camera.move_and_look_at(self.camera_x, self.camera_y, self.camera_z, x, y, 0.6)



class RoboschoolForwardWalkerMujocoXML6(RoboschoolForwardWalker6, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)




class RoboschoolHalfCheetah6(RoboschoolForwardWalkerMujocoXML6):

    foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh", "neck1", "neck2", "head"]  # track these contacts with ground


    def __init__(self):
        #self.EPOCHSIZE = 2000
        #self.log_rewards = np.zeros((self.EPOCHSIZE,7), dtype=np.float32)
        RoboschoolForwardWalkerMujocoXML2.__init__(self, "half_cheetah6.xml", "torso", action_dim=8, obs_dim=33, power=0.90)

    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] and not self.feet_contact[6] and not self.feet_contact[7] and not self.feet_contact[8] else -1

    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML6.robot_specific_reset(self)
        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef  = 90.0
        self.jdict["bfoot"].power_coef  = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef  = 60.0
        self.jdict["ffoot"].power_coef  = 30.0
        self.jdict["neck1_joint1"].power_coef  = 40.0
        self.jdict["neck2_joint1"].power_coef  = 40.0

    def step(self, a):
        #print(self.parts["head"].pose().xyz())
        return RoboschoolForwardWalkerMujocoXML6.step(self, a)
