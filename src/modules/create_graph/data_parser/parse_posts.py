import csv
from ...utils.structures.post import Post


class PostHandler:

    def __init__(self):
        self.posts = []

    @staticmethod
    def is_number(str):
        try:
            return float(str)
        except ValueError:
            return None

    def posts_parse(self, post_path):
        posts = []
        with open(post_path, encoding="utf8") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:
                # address = [row[13], row[14], row[15]]
                address = [row[0]]
                uuid = [row[1]]
                if (self.is_number(row[2]) != None and self.is_number(row[3]) != None):
                    post = Post(' '.join(address),' '.join(uuid), self.is_number(row[2]), self.is_number(row[3]))
                    posts.append(post)
        return posts

    def read_postal_offices(self, post_paths):
        """
        Postal offices are read from csv file and than added to array
        :param post_paths:
        :return:
        """
        self.posts = []
        for name, post_path in post_paths.items():
            self.posts.extend(self.posts_parse(post_path))

        return self.posts
