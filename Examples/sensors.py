
class Sensor:
    def __init__(self, sensors=set(), coverage=dict([]), jamming_actions=dict([]), sensor_noise=0, sensor_cost_dict=dict([])):
        self.sensors = sensors
        self.coverage = coverage
        self.jamming_actions = jamming_actions
        self.sensor_noise = sensor_noise
        self.sensor_cost_dict = sensor_cost_dict

    def set_coverage(self, sensor_con, covered_set):
        # for each sensor, a subset of the surveillance region
        self.coverage[sensor_con] = covered_set
        return

    def get_coverge(self, sensor_con):
        return self.coverage[sensor_con]

    def jam(self):
        return self.jamming_actions
