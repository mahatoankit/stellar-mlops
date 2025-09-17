print("TEST: Module loading started")


def load_config():
    print("TEST: load_config called")
    return {"test": "value"}


print("TEST: Module loading completed")
print("TEST: load_config function defined:", "load_config" in globals())
