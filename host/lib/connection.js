const { USB } = require("webusb");
const { ticalc } = require("ticalc-usb");

const { withTimeout } = require("./protocol");

async function connect() {
  const usb = new USB();
  global.navigator = global.navigator || {};
  global.navigator.usb = usb;

  let connected = null;
  ticalc.addEventListener("connect", (calculator) => {
    connected = calculator;
  });

  await withTimeout(ticalc.init({ usb, supportLevel: "beta" }), 5000, "init");
  if (!connected) {
    await withTimeout(ticalc.choose({ usb }), 5000, "choose");
  }
  if (!connected) {
    throw new Error("No calculator selected");
  }

  return connected;
}

async function close(calc) {
  try {
    if (calc && calc._d && calc._d._device && calc._d._device.opened) {
      await calc._d._device.close();
    }
  } catch (_) {
  }
}

module.exports = {
  connect,
  close,
};
