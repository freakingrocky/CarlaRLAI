"""
Adhering (Sort of) to the insustry standards set bt OpenAI.

Base Code: https://github.com/Sentdex/Carla-RL
"""

# Import Carla Egg
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major, sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    print("Error!!!")
    exit(1)

import carla

# Importing ML Modules
import tensorflow as tf
from tensorflow.keras.applications import ResNet152
import tensorboard


# Importing Other Modules
import random
import time
import math
import cv2
import numpy as np
from collections import deque
from threading import Thread
from tqdm import tqdm


# Global Variables
SHOW_PREVIEW = False
SHOW_DEPTH_PREVIEW = False
SHOW_DVS_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 360
MODEL = 'Cybertruck'
TRANSFORM = carla.Transform(carla.Location(x=2.2, z=2.4))
ACC = 0.4
SPEED = 50
EPISODE_DURATION = 15
# Synchronous Mode
SYNCHRONOUS = False
REPLAY_MEMORY_SIZE = 50_000  # 50_000 = 50,000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 1
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
DELTA_t = 5
FRAME_RATE = 15

# Pre-Made Model
MODEL_NAME = 'ResNet152'

MIN_REWARD = -200
# Epochs
EPISODES = 10  # EPOCHS

DISCOUNT = 0.99
epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

AGGREGATE_STEPS_EVERY = 10


class ModifiedTensorBoard(tf.keras.callbacks.TensorBoard):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def set_model(self, model):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


class Car:
    visualize = SHOW_PREVIEW
    visualize_dvs = SHOW_DVS_PREVIEW
    visualize_depth = SHOW_DEPTH_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    rgb_camera = None
    depth_cam = None
    dvs_cam = None
    model = MODEL
    accelration = ACC
    speed = SPEED
    transform = TRANSFORM
    sync = SYNCHRONOUS
    fr = FRAME_RATE

    def __init__(self):
        # Basic Initializaion, Connecting & Getting some data
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(15.0)
        self.world = self.client.get_world()
        self.blueprint_lib = self.world.get_blueprint_library()
        self.vehicle_model = self.blueprint_lib.filter(self.model)[0]
        self.spawn_camera_location = self.transform
        self.frame_rate = self.fr

        # Synchronous Mode
        if self.sync:
            self.carla_settings = self.world.get_settings()
            self.carla_settings.synchronous_mode = self.sync
            self.carla_settings.fixed_delta_seconds = 1.0 / \
                self.frame_rate if self.frame_rate > 10 else 10
            self.world.apply_settings(self.carla_settings)

    def reset(self):
        self.collisions = []
        self.actor_list = []
        self.lane_events = []

        # Spawning Vehicle
        #! Problem:- Spawn Colliosns every time
        self.position = random.choice(self.world.get_map().get_spawn_points())
        spawned = False
        while not spawned:
            try:
                self.vehicle = self.world.spawn_actor(
                    self.vehicle_model, self.position)
                spawned = True
            except RuntimeError:
                pass

        self.actor_list.append(self.vehicle)

        # Spawning RGB Camera
        self.camera_model = self.blueprint_lib.find("sensor.camera.rgb")
        self.camera_model.set_attribute("image_size_x", f"{self.im_width}")
        self.camera_model.set_attribute("image_size_y", f"{self.im_height}")
        self.camera_model.set_attribute('fov', '160')
        self.sensor_camera = self.world.spawn_actor(
            self.camera_model, self.spawn_camera_location, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.sensor_camera)
        self.sensor_camera.listen(lambda data: self.process_image(data))

        # Spawning Depth Camera
        self.depth_camera = self.blueprint_lib.find("sensor.camera.depth")
        self.depth_camera.set_attribute("image_size_x", f"{self.im_width}")
        self.depth_camera.set_attribute("image_size_y", f"{self.im_height}")
        self.depth_camera.set_attribute('fov', '110')
        self.sensor_depth = self.world.spawn_actor(
            self.depth_camera, self.spawn_camera_location, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.sensor_depth)
        self.sensor_depth.listen(lambda data: self.process_image_depth(data))

        # Spawning DVS Sensor
        self.dvs_sensor = self.blueprint_lib.find("sensor.camera.dvs")
        self.dvs_sensor.set_attribute("image_size_x", f"{self.im_width}")
        self.dvs_sensor.set_attribute("image_size_y", f"{self.im_height}")
        self.dvs_sensor.set_attribute('fov', '110')
        self.dvs_sensor.set_attribute('use_log', 'true')
        self.dvs_sensor.Color = 'Raw'
        self.dvs_sensor.set_attribute('gamma', '2.2')
        self.sensor_dvs = self.world.spawn_actor(
            self.dvs_sensor, self.spawn_camera_location, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.sensor_dvs)
        self.sensor_dvs.listen(lambda data: self.process_image_dvs(data))

        # The car falls from the sky and cannot be interacted with for some time
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        # collision Sensor (Labels)
        self.collision_sensor = self.blueprint_lib.find(
            "sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            self.collision_sensor, self.transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.collision_sensor.listen(lambda event: self.collision_data(event))

        # Lane Invasion Sensor (Labels)
        self.lane_sensor = self.blueprint_lib.find(
            "sensor.other.lane_invasion")
        self.lane_sensor = self.world.spawn_actor(
            self.lane_sensor, self.transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.lane_sensor.listen(lambda event: self.lane_data(event))

        # If the camera is still not ready.
        while any(value is None for value in [self.rgb_camera, self.depth_cam, self.dvs_cam]):
            time.sleep(0.01)

        # Episode Time
        self.episode_start = time.time()
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

        return self.fuse(self.rgb_camera, self.depth_cam, self.dvs_cam)

    def collision_data(self, event):
        self.collisions.append(event)

    def lane_data(self, event):
        self.lane_events.append(event)

    def process_image(self, data):
        IMGArray = np.array(data.raw_data)
        IMG = IMGArray.reshape((self.im_height, self.im_width, 4))
        IMG = IMG[:, :, :3]
        if self.visualize:
            cv2.imshow("Visualization", IMG)
            cv2.waitKey(1)
        self.rgb_camera = IMG/255

    def process_image_depth(self, data):
        IMGArray = np.array(data.raw_data)
        IMG = IMGArray.reshape((self.im_height, self.im_width, 4))
        IMG = IMG[:, :, :3]
        IMG = IMG[:, :, ::-1]
        IMG = ((IMG[:, :, 0] + IMG[:, :, 1]*256.0 +
                IMG[:, :, 2]*256.0*256.0)/((256.0*256.0*256.0) - 1))
        if self.visualize_depth:
            cv2.imshow("Depth", IMG)
            cv2.waitKey(1)
        self.depth_cam = IMG

    def process_image_dvs(self, data):
        Events = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', np.bool)]))
        IMGArray = np.zeros((self.im_height, self.im_width, 3), dtype=np.uint8)
        IMGArray[Events[:]['y'], Events[:]['x'], Events[:]['pol'] * 2] = 255
        IMGArray.swapaxes(0, 1)
        if self.visualize_dvs:
            cv2.imshow("DVS", IMGArray)
            cv2.waitKey(1)
        self.dvs_cam = IMGArray

    def step(self, action):
        # 0 is left, 1 is straight & 2 is right
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=self.accelration, brake=-1*self.STEER_AMT))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=self.accelration, brake=0*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(
                throttle=self.accelration, brake=1*self.STEER_AMT))

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(abs(v.x**2 + v.y**2 + v.z**2)))

        if len(self.collisions):
            done = True
            reward = -200

        if 'solid' in [val.lower() for sublist in self.lane_events for val in sublist]:
            done = True
            reward = -50

        # Avoiding driving in circles
        elif kmh < self.speed:
            done = False
            reward = -1

        else:
            done = True
            reward = 1

        if self.episode_start + EPISODE_DURATION < time.time():
            done = True

        return self.fuse(self.rgb_camera, self.depth_cam, self.dvs_cam), reward, done, None

    def fuse(self, rgb_cam, depth_cam, dvs_cam):
        # ! TODO, Sensor Fusion (RGB-D + Event Based)
        # Current Idea:
        #     Spiking

        # Simplest form of RGB-D fusion. Simply added another channel
        data = np.hstack((rgb_cam, depth_cam))

        return data


class DQNAgent:
    def __init__(self):
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        # Limit the no. of outputs to tensorboard
        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")
        self.target_update_counter = 0

        self.terminate = False
        self.last_logged_episode = 0
        self.training_init = False

    def create_model(self):
        base_model = ResNet152(
            weights=None, include_top=False, input_shape=(IM_HEIGHT, IM_WIDTH, 4)) 

        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # x = tf.keras.layers.GlobalMaxPooling2D()(x)

        predictions = tf.keras.layers.Dense(3, activation="relu")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

        Admax_Optim = tf.keras.optimizers.Adamax(learning_rate=0.001)

        model.compile(loss="mse",   # Mean Squared Error
                      optimizer=Admax_Optim,
                      metrics=["accuracy"])

        return model

    def update_replay_memory(self, transition):
        # transition = (current_state, action, reward, new_state, done)
        self.replay_memory.append(transition)

    def train(self):
        """The Q-Learning Algorithm."""
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return False

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs = self.model.predict(
            current_states, PREDICTION_BATCH_SIZE)

        future_states = np.array([transition[3] for transition in minibatch])
        future_qs = self.target_model.predict(
            current_states, PREDICTION_BATCH_SIZE)

        X = []
        y = []

        # If it is a terminal state
        for index, (current_state, action, reward, new_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

        current_Qs = current_qs[index]
        current_Qs[action] = new_q

        X.append(current_state)
        y.append(current_Qs)

        log_this_step = False
        if self.tensorboard.step > self.last_logged_episode:
            log_this_step = True
            self.last_log_episode = self.tensorboard.step

        self.model.fit(np.array(X), np.array(y), batch_size=TRAINING_BATCH_SIZE, verbose=0,
                       shuffle=False, callbacks=[self.tensorboard] if log_this_step else None)

        if log_this_step:
            self.target_update_counter += 1

        if self.target_update_counter > DELTA_t:
            # Basically copying the working model.
            self.target_model.set_weights(self.model.get_weights())

    def get_qs(self, state):
        return self.model.predict(np.array(state).reshape(-1, *state.shape))[0]

    def train_loop(self):
        # Train & Predict in seperate threads.
        X = np.random.uniform(
            size=(1, IM_HEIGHT, IM_WIDTH, 3)).astype(np.float32)
        y = np.random.uniform(size=(1, 3)).astype(np.float32)

        self.model.fit(X, y, batch_size=1, verbose=False)

        self.training_init = True

        while True:
            if self.terminate:
                return
            self.train()
            time.sleep(0.001)


if __name__ == '__main__':
    try:
        FPS = 15  # This is only possible due to async mode runtime
        ep_rewards = [-200]

        # tf.config.optimizer.set_jit(True)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # To Allow for replication of the results
        random.seed(1)
        np.random.seed(1)
        tf.random.set_seed(1)

        if not os.path.isdir("Models"):
            os.makedirs("Models")

        agent = DQNAgent()
        env = Car()

        trainer = Thread(target=agent.train_loop, daemon=True)
        trainer.start()

        while not agent.training_init:
            time.sleep(0.01)

        agent.get_qs(np.ones((env.im_height, env.im_width, 3)))

        for episode in tqdm(range(1, EPISODES+1), ascii=True, unit="Episodes"):
            env.collisions = []
            env.lane_events = []
            agent.tensorboard.step = episode
            episode_reward = 0
            step = 1
            current_state = env.reset()
            done = False
            episode_start = time.time()

            # while not done:
            while True:
                if np.random.random() > epsilon:
                    action = np.argmax(agent.get_qs(current_state))
                else:
                    action = np.random.randint(0, 3)
                    time.sleep(1 / FPS)  # Analogous to 1/dt in games

                new_state, reward, done, _ = env.step(action)
                episode_reward += reward
                agent.update_replay_memory(
                    (current_state, action, reward, new_state, done))

                step += 1

                if done:
                    break

            for actor in env.actor_list:
                actor.destroy()

            ep_rewards.append(episode_reward)
            if not episode % AGGREGATE_STEPS_EVERY or episode == 1:
                tmp = ep_rewards[-AGGREGATE_STEPS_EVERY:]
                average_reward = sum(tmp)/len(tmp)
                min_reward = min(tmp)
                max_reward = max(tmp)
                agent.tensorboard.update_stats(
                    reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward)

                if average_reward >= -100:
                    agent.model.save(
                        f"Models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model")
                else:
                    if os.path.exists("Models/latest.model"):
                        os.remove("Models/latest.model")
                    agent.model.save("Models/latest.model")

            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)

        agent.terminate = True
        trainer.join()
        agent.model.save(
            f"Models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model")
    finally:
        print("Cleaning Up..")
        for actor in env.actor_list:
            actor.destroy()
        print("Done.")
