from titanic import Titanic


def main():
    titanic = Titanic("gmodel2", epochs=50, lr=0.001, drop_out=0.45)

    titanic.run_me()


if __name__ == "__main__":
    main()
