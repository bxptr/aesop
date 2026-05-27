const {
  VPKT_DELAY_ACK,
  VPKT_ERROR,
  USB_ERROR_CODES,
  USB_ERROR_MESSAGES,
  withTimeout,
  intBE,
} = require("./protocol");

function destructRawPacket(data) {
  return {
    size: intBE(data.slice(0, 4)),
    type: data[4],
    data: data.slice(5, 5 + intBE(data.slice(0, 4))),
  };
}

function destructVirtualPacket(data) {
  return {
    size: intBE(data.slice(0, 4)),
    type: intBE(data.slice(4, 6)),
    data: data.slice(6, 6 + intBE(data.slice(0, 4))),
  };
}

async function recvRawPacket(calc) {
  const endpoint = calc._d._inEndpoint;
  const length = Math.max(calc._d._bufferSize || 0, endpoint.packetSize);
  const result = await calc._d._device.transferIn(endpoint.endpointNumber, length);
  if (result.status !== "ok") {
    throw new Error(`Error receiving data from device: ${result.status}`);
  }

  return new Uint8Array(result.data.buffer, result.data.byteOffset, result.data.byteLength);
}

async function recvAnyVirtual(calc) {
  const raw = destructRawPacket(await recvRawPacket(calc));
  if (raw.type !== 4) {
    throw new Error(`expected final raw virtual packet, got raw ${raw.type}`);
  }

  const packet = destructVirtualPacket(raw.data);
  await calc._d._sendAck();
  return packet;
}

async function recvVirtualPacket(calc) {
  const chunks = [];

  for (;;) {
    const raw = destructRawPacket(await recvRawPacket(calc));
    if (raw.type !== 3 && raw.type !== 4) {
      throw new Error(`expected raw virtual data, got raw ${raw.type}`);
    }
    chunks.push(Buffer.from(raw.data));
    await calc._d._sendAck();
    if (raw.type === 4) {
      break;
    }
  }

  return destructVirtualPacket(Buffer.concat(chunks));
}

async function expectVirtual(calc, expectedType, label) {
  const packet = await withTimeout(recvVirtualPacket(calc), 60000, label);
  if (packet.type === VPKT_DELAY_ACK) {
    const delayMs = intBE(packet.data);
    await new Promise((resolve) => setTimeout(resolve, Math.max(1, delayMs / 1000)));
    return expectVirtual(calc, expectedType, label);
  }
  if (packet.type === VPKT_ERROR) {
    const code = intBE(packet.data.slice(0, 2));
    const index = USB_ERROR_CODES.indexOf(code);
    const message = index >= 0 ? USB_ERROR_MESSAGES[index] : "unknown calculator error";
    throw new Error(`${label}: calculator error 0x${code.toString(16).padStart(4, "0")} (${message})`);
  }
  if (packet.type !== expectedType) {
    throw new Error(`${label}: expected 0x${expectedType.toString(16)}, got 0x${packet.type.toString(16)}`);
  }
  return packet;
}

module.exports = {
  recvAnyVirtual,
  recvVirtualPacket,
  expectVirtual,
};
