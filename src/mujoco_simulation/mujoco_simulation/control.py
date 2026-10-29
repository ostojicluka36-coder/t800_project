import rclpy
import numpy as np
from rclpy.node import Node

class ControlLaw(Node):

    def __init__(self, pd_gains=0.0):
        self.pd_gains = pd_gains



def main(args=None):
    pass

if __name__ == '__main__':
    main()