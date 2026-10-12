"""Wheel-speed waypoint tracker with damped heading trim."""
import math

import numpy as np


class Controller:
    def reset(self, info: dict)->None:
        self.waypoints=[(0.35,0.55),(-0.20,0.55),(-0.90,0.55)]
        self.index=0; self.radius=0.075

    def step(self, obs: dict[str,float], t: float)->dict[str,float]:
        x,y=obs["robot_x"],obs["robot_y"]
        while self.index<len(self.waypoints)-1 and math.hypot(self.waypoints[self.index][0]-x,self.waypoints[self.index][1]-y)<0.16:
            self.index+=1
        gx,gy=self.waypoints[self.index]; dx,dy=gx-x,gy-y
        distance=math.hypot(dx,dy); desired=math.atan2(dy,dx)
        error=(desired-obs["joint.yaw.pos"]+math.pi)%(2.0*math.pi)-math.pi
        forward=min(0.55,1.1*distance)*max(0.12,math.cos(error))
        wheel_speed=forward/self.radius
        left=float(np.clip(0.50*(wheel_speed-obs["joint.left_wheel_spin.vel"]),-2.5,2.5))
        right=float(np.clip(0.50*(wheel_speed-obs["joint.right_wheel_spin.vel"]),-2.5,2.5))
        yaw_rate=obs["joint.yaw.vel"]
        trim=float(np.clip(2.0*error-0.45*yaw_rate,-3.0,3.0))
        return {"left_drive":left,"right_drive":right,"heading_trim":trim}
