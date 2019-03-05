import random
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFilter
from PIL import ImageFont


def randomChar():
    '''
    随机生成chr
    :return:返回一个随机生成的chr
    '''
    return chr(random.randint(48, 57))


def randomBgColor():
    '''
    随机生成验证码的背景色
    :return:
    '''
    return (random.randint(50, 100), random.randint(50, 100), random.randint(50, 100))


def randomTextColor():
    '''
    随机生成验证码的文字颜色
    :return:
    '''
    return (random.randint(120, 200), random.randint(120, 200), random.randint(120, 200))


w = 30 * 4
h = 60

# 设置字体类型及大小
font = ImageFont.truetype(font=r'/home/tensorflow01/oneday/seq2seq验证码识别/NotoSansCJK-Black.ttc', size=36)

for _ in range(500):
    # 创建一张图片，指定图片mode，长宽
    image = Image.new('RGB', (w, h), (255, 255, 255))

    # 创建Draw对象
    draw = ImageDraw.Draw(image)
    # 遍历给图片的每个像素点着色
    for x in range(w):
        for y in range(h):
            draw.point((x, y), fill=randomBgColor())

    # 将随机生成的chr，draw如image
    filename = []
    for t in range(4):
        ch = randomChar()
        filename.append(ch)
        draw.text((30 * t, 10), ch, font=font, fill=randomTextColor())

    # 设置图片模糊
    # image = image.filter(ImageFilter.BLUR)
    # 保存图片
    image_path = r"/home/tensorflow01/oneday/seq2seq验证码识别/验证码图片"
    image.save("{0}/{1}.jpg".format(image_path, "".join(filename)))
