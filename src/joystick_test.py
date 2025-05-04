import hid
import time

VENDOR_ID  = 0x054c
PRODUCT_ID = 0x05c4

def normalize(raw):
    return (raw - 128) / 128.0

def main():

    for device in hid.enumerate():
        # print(device)
        print(f"0x{device['vendor_id']:04x}:0x{device['product_id']:04x} {device['usage_page']:04x}:{device['usage']:04x} {device['product_string']}")

    # Controllers have usage page 0x01 and usage 0x05

    gamepad = hid.device()
    gamepad.open(VENDOR_ID, PRODUCT_ID)
    gamepad.set_nonblocking(True)

    try:
        while True:
            rpt = gamepad.read(64)
            if rpt:
                lx, ly, rx, ry = map(normalize, rpt[1:5])
                print(f"L=({lx:.2f},{ly:.2f}) R=({rx:.2f},{ry:.2f})")
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        gamepad.close()

if __name__ == "__main__":
    main()
