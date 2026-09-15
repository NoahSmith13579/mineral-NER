from corpus.builder import create_corpus
from server import start_server
from training.grid_search import grid_search


def main():
    create_corpus()
    grid_search()
    start_server()


if __name__ == "__main__":
    main()
