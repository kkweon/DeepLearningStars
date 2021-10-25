# pylint: disable=C0103
"""
Multiprocess version of list2md.py


Examples
----------

    $ python list2md.multiprocess.py

"""
import argparse
import time
import logging
import sys

from multiprocessing.pool import Pool
from typing import Iterator, List

import requests
from mypy_extensions import TypedDict

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--token", help="GitHub API Token to query repo stars.")
parser.add_argument("--workers", help="The number of workers to fetch GitHub stars.")
args = parser.parse_args()


GitType = TypedDict(
    "GitType",
    {
        "name": str,
        "url": str,
        "stars": int,
        "description": str,
        "created": str,
        "updated": str,
        "forks": str,
    },
)


def write_md(dict_list: List[GitType], filepath: str = "README.md") -> bool:
    """Given a list of dict, write a markdown file

    Parameters
    ----------
    dict_list : List[GitType]

        [row1, row2, ...]

        where row1 = {
            "name": "Tensorflow",
            "url": "https://github.com/tensorflow/tensorflow",
            "stars": 55359,
            "description": "Computation using data flow graph ..."
        }

    filepath : str
        Readme path

    Returns
    ----------
    bool
        Returns True If everything went smooth
    """

    head = (
        "# Top Deep Learning Projects\n"
        "A list of popular github projects ordered by stars.\n"
        "Please update list.txt (via pull requests)\n\n"
        "|Project Name| Stars | Description |\n"
        "| ---------- |:-----:| ----------- |\n"
    )

    tail = "\n\nLast Automatic Update: {}\n".format(time.strftime("%c (%Z)"))

    # sort descending by n_stars
    dict_list = sorted(dict_list, key=lambda x: x["stars"], reverse=True)

    # each data is a string (see `dict2md`)
    data_list = map(dict2md, dict_list)

    with open(filepath, "w") as out:

        out.write(head)
        out.write("\n".join(data_list))
        out.write(tail)

        return True

    return False


def dict2md(dict_: GitType) -> str:
    """Convert a dictionary to a markdown format"""
    return "| [{name}]({url}) | {stars} | {description} |".format(**dict_)


def get_url_list(filepath: str = "list.txt") -> Iterator[str]:
    """Read list.txt and returns a list of API urls"""

    def preprocess_url(url: str) -> str:
        """Returns an API url"""
        return "https://api.github.com/repos/{}".format(url[19:].strip().strip("/"))

    with open(filepath, "r") as file:
        data = file.readlines()

    return map(preprocess_url, data)


def grab_data(url: str) -> GitType:
    """Go to the URL and grab a data

    Parameters
    ----------
    url : str
        URL to a github repo

    Returns
    ----------
    dict
        dict_keys(['name',
                   'description',
                   'forks',
                   'created',
                   'updated',
                   'url',
                   'stars'])
    """
    headers = {"Authorization": "token " + args.token}

    try:
        print("Accessing to {}".format(url))
        data_dict = requests.get(url, headers=headers).json()

        return {
            "name": data_dict["name"],
            "description": data_dict["description"],
            "forks": data_dict["forks_count"],
            "created": data_dict["created_at"],
            "updated": data_dict["updated_at"],
            "url": data_dict["html_url"],
            "stars": data_dict["stargazers_count"],
        }

    except KeyError:
        raise Exception("Failed to grab data for {}, got response = {}".format(url, data_dict))


def main() -> None:
    """Main function"""
    if not args.token:
        logging.error(
            "GitHub Token is missing. Please pass your GitHub token key as a --token=xxxxxx"
        )
        sys.exit(1)
        return

    url_list = get_url_list()

    pool = Pool(processes=args.workers)
    result = pool.map_async(grab_data, url_list)

    write_md(result.get())


if __name__ == "__main__":
    main()
