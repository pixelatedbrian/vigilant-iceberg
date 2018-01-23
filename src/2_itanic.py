from titanic_v2 import Titanic
import numpy as np


def main():
    for x in range(40):

        lr = np.random.uniform(0.00001, 0.0075)
        drop_out = np.random.uniform(0.15, 0.60)

        batches = [16, 24, 32, 48, 64]
        batch_size = batches[np.random.randint(0, len(batches) - 1)]
        lr_decay = np.random.uniform(-10, -3.3)
        lr_decay = 4**lr_decay

        print("\n\nlr{:0.4f} do{:0.3f} bs{:3d} lrd{:0.4f}\n\n".format(lr, drop_out, batch_size, lr_decay))

        titanic = Titanic(model_name="model2",
                          epochs=100,
                          lr=lr,
                          drop_out=drop_out,
                          batch_size=batch_size,
                          lr_decay=lr_decay,
                          augment_rotate=False,
                          augment_ud=True,
                          c3_transform=1)

        # titanic = Titanic(model_name="model2",
        #                   epochs=50,
        #                   lr=0.00125,
        #                   drop_out=0.30,
        #                   batch_size=64,
        #                   lr_decay=1e-6,
        #                   augment_rotate=False,
        #                   augment_ud=True,
        #                   c3_transform=1)

        titanic.run_me()


if __name__ == "__main__":
    main()
