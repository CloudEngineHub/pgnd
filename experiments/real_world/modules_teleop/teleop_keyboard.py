import threading
import multiprocess as mp
import time

import numpy as np
from pynput import keyboard

from modules_planning.udp_util import udpReceiver, udpSender
from modules_planning.common.communication import XARM_STATE_PORT, XARM_CONTROL_PORT_L
from modules_planning.common.xarm import GRIPPER_OPEN_MIN, GRIPPER_OPEN_MAX, POSITION_UPDATE_INTERVAL, COMMAND_CHECK_INTERVAL

np.set_printoptions(precision=2, suppress=True)

class KeyboardTeleop(mp.Process):
    
    XYZ_STEP = 5
    XYZ_STEP_STOW = 1
    ANGLE_STEP = 1
    ANGLE_STEP_STOW = 0.2
    GRIPPER_STEP = 100
    GRIPPER_STEP_STOW = 20

    def __init__(
        self, 
        gripper_enable=False,
    ) -> None:
        super().__init__()
        self.gripper_enable = gripper_enable

        self.key_states = {
            "w": False,
            "s": False,
            "a": False,
            "d": False,
            "up": False,    # Up arrow key for moving up
            "down": False,  # Down arrow key for moving down
            "left": False,
            "right": False,
            
            "i": False,
            "k": False,
            "j": False,
            "l": False,
            "u": False,
            "o": False,

            "[": False,
            "]": False,
            "{": False,
            "}": False,

            ",": False,
            ".": False,

            "p": False,
            "r": False,  # Key for reset the position
            "space": False,  # Key for send current position
            "shift": False,  # Key for slow down the speed
        }

        # additional states
        self.init = True
        self.pause = False
        self.record_start = mp.Value('b', False)
        self.record_stop = mp.Value('b', False)

        # states for moving the arm
        self.cur_position = None
        self.command = []

        self._alive = mp.Value('b', True)
        self.controller_quit = False

        self.state_receiver = None # udpReceiver(ports={'xarm_state':XARM_STATE_PORT})
        self.command_sender = None # udpSender(port=XARM_CONTROL_PORT)

    @staticmethod
    def log(msg):
        print(f"\033[94m{msg}\033[0m")

    def on_press(self, key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = True
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = True
            elif key == keyboard.Key.down:
                self.key_states["down"] = True
            elif key == keyboard.Key.left:
                self.key_states["left"] = True
            elif key == keyboard.Key.right:
                self.key_states["right"] = True
            elif key == keyboard.Key.space:
                self.key_states["space"] = True
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = True

    def on_release(self, key):
        try:
            key_char = key.char.lower() if key.char else key.char
            if key_char in self.key_states:
                self.key_states[key_char] = False
        except AttributeError:
            if key == keyboard.Key.up:
                self.key_states["up"] = False
            elif key == keyboard.Key.down:
                self.key_states["down"] = False
            elif key == keyboard.Key.left:
                self.key_states["left"] = False
            elif key == keyboard.Key.right:
                self.key_states["right"] = False
            elif key == keyboard.Key.space:
                self.key_states["space"] = False
            elif key == keyboard.Key.shift:
                self.key_states["shift"] = False
            
            
            if key == keyboard.Key.esc:
                return False

    def update_xarm_pos(self):
        self.state_receiver = udpReceiver(ports={'xarm_state': XARM_STATE_PORT}, re_use_address=True)
        self.state_receiver.start()
        
        while self.alive:
            xarm_state = self.state_receiver.get("xarm_state", pop=True)
            if xarm_state is not None:
                self.cur_position = xarm_state["pos"]
                if self.gripper_enable:
                    self.cur_gripper_pos = xarm_state["gripper"]
            time.sleep(POSITION_UPDATE_INTERVAL / 10)

        self.state_receiver.stop()
        self.log(f"update_xarm_pos exit!")

    def get_command(self):
        if self.cur_position is None:
            return
            
        x, y, z, roll, pitch, yaw = self.cur_position
        
        if self.key_states["p"]:
            # abandon all other keyinputs
            self.pause = not self.pause
            self.log(f"keyboard teleop pause: {self.pause}")
            time.sleep(0.5)

        if self.key_states[","]:
            self.record_start.value = True
            self.log(f"Record start")
            time.sleep(0.5)
        
        if self.key_states["."]:
            self.record_stop.value = True
            self.log(f"Record stop")
            time.sleep(0.5)

        if self.pause:
            self.command = []
            return
        else:
            xyz_step = self.XYZ_STEP_STOW if self.key_states["shift"] else self.XYZ_STEP
            angle_step = self.ANGLE_STEP_STOW if self.key_states["shift"] else self.ANGLE_STEP
            gripper_step = self.GRIPPER_STEP_STOW if self.key_states["shift"] else self.GRIPPER_STEP
            gripper_changed = False
            if self.key_states["w"]:
                x += xyz_step
            if self.key_states["s"]:
                x -= xyz_step
            if self.key_states["a"]:
                y += xyz_step
            if self.key_states["d"]:
                y -= xyz_step
            if self.key_states["up"]:
                z += xyz_step   # Move up
            if self.key_states["down"]:
                z -= xyz_step   # Move down
            
            # adjusting the camera orientation
            if self.key_states["k"]:
                pitch = (pitch + angle_step) % 360
            if self.key_states["i"]:
                pitch = (pitch - angle_step) % 360
            if self.key_states["u"]:
                yaw = (yaw + angle_step) % 360
            if self.key_states["o"]:
                yaw = (yaw - angle_step) % 360
            if self.key_states["j"]:
                roll = (roll + angle_step) % 360
            if self.key_states["l"]:
                roll = (roll - angle_step) % 360

            if self.key_states["["]:
                gripper_pos -= gripper_step
                gripper_changed = True
            if self.key_states["]"]:
                gripper_pos += gripper_step
                gripper_changed = True
            if self.key_states["{"]:
                gripper_pos -= gripper_step
                gripper_changed = True
            if self.key_states["}"]:
                gripper_pos += gripper_step
                gripper_changed = True
            if self.key_states["left"]:
                gripper_pos = GRIPPER_OPEN_MIN
                gripper_changed = True
            elif self.key_states["right"]:
                gripper_pos = GRIPPER_OPEN_MAX
                gripper_changed = True

            controller_quit = False
            if self.key_states["space"]:
                # if space key is pressed, send current position
                controller_quit = True
            self.controller_quit = controller_quit

            if self.key_states["r"]:
                self.command = ["reset"]
                return

            if self.gripper_enable and gripper_changed:
                next_position = [x, y, z, roll, pitch, yaw, gripper_pos]
            else:
                next_position = [x, y, z, roll, pitch, yaw]
            
            if any([self.key_states[x] for x in ["w", "s", "a", "d", "up", "down",
                                                 "i", "k", "j", "l", "u", "o",
                                                 "[", "]", "{", "}", "left", "right", 
                                                 "space"]]):
                self.command = [next_position]
            else:
                self.command = []
            return

    def run(self) -> None:
        self.keyboard_listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.keyboard_listener.start()
        
        self.update_pos_t = threading.Thread(name="get_pos_from_XarmController", target=self.update_xarm_pos)
        self.update_pos_t.start()

        self.command_sender = udpSender(port=XARM_CONTROL_PORT_L)

        time.sleep(1)

        while self.alive:
            try:
                self.get_command()
                self.command_sender.send(self.command)
                time.sleep(COMMAND_CHECK_INTERVAL / 2)
            except:
                self.log(f"keyboard teleop error")
                break

        self.stop(self.controller_quit)
        self.command_sender.close()
        self.keyboard_listener.stop()
        self.update_pos_t.join()
        self.log(f"{'='*20} keyboard teleop exit!")
    
    @property
    def alive(self):
        alive = self._alive.value & self.keyboard_listener.is_alive()
        self._alive.value = alive
        return alive 

    def stop(self, stop_controller=False):
        if stop_controller:
            self.log("teleop stop controller")
            self.command_sender.send(["quit"])
            time.sleep(1)
        self.log("teleop stop")
        self._alive.value = False

