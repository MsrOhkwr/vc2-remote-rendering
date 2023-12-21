import cv2
import numpy as np

from app.render import Context

ctx = Context()
ctx.bind_data(env_map_path="tests/data/test_env_map.hdr")
ctx.create_program()
ctx.render(1)
buffer = ctx.read_buffer(Context.ATTACHMENT_INDEX_OUTPUT_COLOR)
buffer = np.flipud(buffer)
buffer = cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGRA)
buffer = (buffer * 255).astype(np.uint8)
cv2.imwrite("tests/data/reference.jpg", buffer)
