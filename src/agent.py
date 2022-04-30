from drqn import Q_Network
from constructor import State, Action, Obs, PomdpInit

class Agent:
    def __init__(self, pomdp):
        # init state
        self.current_state = pomdp.get_state(False, {'object': '', 'person': ''}, self.get_obj_list(pomdp))

    def get_person(self):
        print("Who am I talking to?")

    @staticmethod
    def get_obj_list(pomdp):
        obj = ''
        for i in range(len(pomdp._known_props)):
            obj += '0'
        return [obj, obj, obj]

    def policy(self, state):
        # TODO
        return