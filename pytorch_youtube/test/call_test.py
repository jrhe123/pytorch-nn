from typing import Any


class Person:

    def __call__(self, name):
        print("__call__" + name)

    def hello(self, name):
        print("hello" + name)


p = Person()
p("zhangsan")
p.hello("lisi")
