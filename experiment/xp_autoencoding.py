from keras.callbacks import ModelCheckpoint

from models.vae import VAE
from grund.match import MatchConfig, Match


def data_stream(batch_size):
    cfg = MatchConfig(canvas_size=(64, 64), players_per_side=2, random_initialization=True,
                      ball_pixel_radius=3, players_pixel_radius=6)
    env = Match(cfg)
    while 1:
        state, reward = env.random_state(batch_size)
        state = state / 255. - 0.5
        yield state, state


network = VAE(input_image_shape=(64, 64, 3), encoded_output_dim=32)
network.model.fit_generator(data_stream(32), steps_per_epoch=1000, epochs=100,
                            callbacks=[ModelCheckpoint("vae.model")])
