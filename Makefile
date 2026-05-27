CEDEV := $(CURDIR)/tools/CEdev
HOST := node host/ti84-webusb.js
PY := python3

export PATH := $(CEDEV)/bin:$(PATH)

.PHONY: build send info dir out eval parity export check clean

build:
	$(MAKE) -C runtime NAME=AESOP

send: build
	@ls runtime/bin/AESOP.8xp.*.8xv 2>/dev/null | sort | while read file; do \
		$(HOST) send "$$file"; \
	done
	$(HOST) send runtime/bin/AESOP.8xp

info:
	$(HOST) info

dir:
	$(HOST) dir

out:
	$(HOST) appvar GENOUT

eval:
	$(PY) -m aesop.eval_sampler --seeds 32 --show 3

parity:
	$(PY) -m compiler.parity

export:
	$(PY) -m compiler.export --checkpoint aesop/checkpoints/model.npz --out runtime/src/generated_model.h --name token-rnn128-neural-compact2-v384-lat48r4-s4500

check:
	$(PY) -m py_compile aesop/*.py compiler/*.py
	node --check host/ti84-webusb.js
	@for file in host/lib/*.js; do node --check "$$file"; done

clean:
	$(MAKE) -C runtime clean
