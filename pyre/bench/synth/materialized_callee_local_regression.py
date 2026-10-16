def callee(i):
    local_value = i + 1
    raise RuntimeError


def main():
    bound = 0
    for i in range(20000):
        try:
            callee(i)
        except RuntimeError as exc:
            frame_locals = exc.__traceback__.tb_next.tb_frame.f_locals
            if frame_locals.get("local_value") == i + 1:
                bound += 1
    print(f"{bound}/4000")


main()
