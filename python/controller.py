import numpy as np

class Controller():
    def __init__(self, k_pos, k_vel, k_att):
        self.k_pos = k_pos
        self.k_vel = k_vel
        self.k_att = k_att

    def control(self, x, x_c):
        v_c = x_
