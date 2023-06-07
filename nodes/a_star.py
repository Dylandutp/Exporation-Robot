#!/usr/bin/env python3
import rospy
import actionlib
import numpy as np
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseGoal, MoveBaseAction
from nav_msgs.msg import OccupancyGrid
from random import randrange
import time
from heapq import heappop, heappush


class Explore:
    def __init__(self):
        self.rate = rospy.Rate(1)
        self.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        self.move_base.wait_for_server(rospy.Duration(5.0))
        rospy.logdebug("move_base is ready")

        self.x = 0
        self.y = 0
        self.completion = 0

        self.map = OccupancyGrid()
        self.sub_map = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.count = 0
        time.sleep(8)

    def map_callback(self, data):
        valid = False

        while valid is False:
            map_size = randrange(len(data.data))
            self.map = data.data[map_size]

            edges = self.check_neighbors(data, map_size)
            if self.map != -1 and self.map <= 0.2 and edges is True:
                valid = True

        row = map_size / 384
        col = map_size % 384

        self.x = col * 0.05 - 10
        self.y = row * 0.05 - 10

        if self.completion % 2 == 0:
            self.completion += 1
            self.set_goal_position(data)

    def set_goal_position(self, data):
        row, col = self.get_random_valid_position(data)
        self.x_goal = col * 0.05 - 10
        self.y_goal = row * 0.05 - 10

        # Start the robot moving toward the goal
        self.set_goal()

    def get_random_valid_position(self, data):
        valid = False

        while not valid:
            row = randrange(data.info.height)
            col = randrange(data.info.width)

            map_size = row * data.info.width + col
            map_value = data.data[map_size]

            if (
                map_value != -1
                and map_value <= 0.2
                and self.check_neighbors(data, map_size)
            ):
                valid = True

        return row, col

    def set_goal(self):
        rospy.logdebug("Setting goal")

        goal = MoveBaseGoal()

        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.x_goal
        goal.target_pose.pose.position.y = self.y_goal
        goal.target_pose.pose.orientation.w = 1.0

        path = self.plan_path_astar()

        if path:
            rospy.logdebug(f"Path: {path}")
            self.move_along_path(path, goal)
        else:
            rospy.loginfo("No valid path found")

    def move_along_path(self, path, goal):
        for point in path:
            goal.target_pose.pose.position.x = point[0]
            goal.target_pose.pose.position.y = point[1]
            rospy.logdebug(
                f"Moving to goal: {goal.target_pose.pose.position.x, goal.target_pose.pose.position.y}"
            )
            self.move_base.send_goal(goal, self.goal_status)
            self.move_base.wait_for_result()

    def goal_status(self, status, result):
        self.completion += 1

        if status == GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal succeeded")

        if status == GoalStatus.ABORTED:
            rospy.loginfo("Goal aborted")

        if status == GoalStatus.REJECTED:
            rospy.loginfo("Goal rejected")

    def check_neighbors(self, data, map_size):
        """Checks neighbors for random points on the map."""
        unknowns = 0
        obstacles = 0

        for x in range(-3, 4):
            for y in range(-3, 4):
                row = x * data.info.width + y
                try:
                    if data.data[map_size + row] == -1:
                        unknowns += 1
                    elif data.data[map_size + row] > 0.65:
                        obstacles += 1
                except IndexError:
                    pass

        if unknowns > 0 and obstacles < 2:
            return True
        else:
            return False

    def plan_path_astar(self):
        start = (self.x, self.y)
        goal = (self.x_goal, self.y_goal)

        open_set = []
        closed_set = set()

        g_scores = {start: 0}
        f_scores = {start: self.heuristic(start, goal)}
        parents = {}

        heappush(open_set, (f_scores[start], start))

        while open_set:
            _, current = heappop(open_set)

            if current == goal:
                path = [current]
                while current in parents:
                    current = parents[current]
                    path.append(current)
                path.reverse()
                return path

            closed_set.add(current)

            neighbors = self.get_neighbors(current)

            for neighbor in neighbors:
                neighbor_g_score = g_scores[current] + self.distance(current, neighbor)

                if neighbor in closed_set and neighbor_g_score >= g_scores.get(
                    neighbor, float("inf")
                ):
                    continue

                if neighbor_g_score < g_scores.get(neighbor, float("inf")):
                    parents[neighbor] = current
                    g_scores[neighbor] = neighbor_g_score
                    f_scores[neighbor] = neighbor_g_score + self.heuristic(
                        neighbor, goal
                    )

                    if neighbor not in closed_set:
                        heappush(open_set, (f_scores[neighbor], neighbor))

        # No valid path found
        return []

    def heuristic(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def get_neighbors(self, point):
        x, y = point
        neighbors = []

        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy

            if self.is_valid_point(new_x, new_y):
                neighbors.append((new_x, new_y))

        return neighbors

    def is_valid_point(self, x, y):
        if x < 0 or x >= self.map.info.width or y < 0 or y >= self.map.info.height:
            return False

        map_size = y * self.map.info.width + x
        map_value = self.map.data[map_size]

        return (
            map_value != -1
            and map_value <= 0.2
            and self.check_neighbors(self.map, map_size)
        )


def main():
    rospy.init_node("explore", log_level=rospy.DEBUG)
    Explore()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except:
        rospy.ROSInterruptException
