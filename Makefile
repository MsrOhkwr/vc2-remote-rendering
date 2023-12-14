PYTHON_FILES=$(shell \
	find . \
	-path "./venv" -prune -o \
	-type f -name "*.py" \
	-exec echo -n {}" " \; \
)

GLSL_FILES=$(shell \
	find . \
	-path "./venv" -prune -o \
	-type f -name "*.glsl" \
	-exec echo -n {}" " \; \
)

black:
	black ${PYTHON_FILES}

black-check:
	black --check ${PYTHON_FILES}

clang-format:
	clang-format -i ${GLSL_FILES}

clang-format-check:
	clang-format -dry-run -Werror ${GLSL_FILES}

pyflakes:
	pyflakes ${PYTHON_FILES}

notebook:
	python colab/notebook.py > colab/websocket_server.ipynb

notebook-check:
	python colab/notebook.py | diff colab/websocket_server.ipynb -

one-by-one-push:
	python one_by_one_push.py

reference:
	python update_reference.py
