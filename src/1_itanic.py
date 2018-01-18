from titanic import Titanic


def main():
    titanic = Titanic("gmodel2", epochs=5)

    titanic.run_me()


if __name__ == "__main__":
    main()
