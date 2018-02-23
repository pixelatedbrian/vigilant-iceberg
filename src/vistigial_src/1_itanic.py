from titanic import Titanic


def main():
    titanic = Titanic(model_name="model5",
                      epochs=100,
                      lr=0.0001,
                      drop_out=0.2,
                      batch_size=32,
                      lr_decay=0,
                      augment_rotate=False,
		      augment_ud=True,
		              c3_transform=0)

    titanic.run_me()


if __name__ == "__main__":
    main()
