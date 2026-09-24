import gpiod
import time

LED_PINS = [22, 27, 17, 23, 24]  # 엄지=22, 검지=27, 중지=17, 약지=23, 소지=24

chip = gpiod.Chip('/dev/gpiochip0')
config = gpiod.LineSettings(
    direction=gpiod.line.Direction.OUTPUT,
    output_value=gpiod.line.Value.ACTIVE  # 시작하자마자 켜짐
)
led_request = chip.request_lines(
    config={tuple(LED_PINS): config},
    consumer="led_always_on"
)

print("LED 5개 전부 켜짐. Ctrl+C로 종료.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n종료 중... LED는 켜진 채로 유지됩니다.")
    # LED를 끄고 싶으면 아래 for문 주석 해제
    # for pin in LED_PINS:
    #     led_request.set_value(pin, gpiod.line.Value.INACTIVE)
    led_request.release()
