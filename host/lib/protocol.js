const VPKT_EXECUTE = 0x0011;
const VPKT_DATA_ACK = 0xaa00;
const VPKT_PARM_REQ = 0x0007;
const VPKT_PARM_DATA = 0x0008;
const VPKT_DIR_REQ = 0x0009;
const VPKT_RTS = 0x000b;
const VPKT_VAR_HDR = 0x000a;
const VPKT_VAR_REQ = 0x000c;
const VPKT_VAR_CNTS = 0x000d;
const VPKT_MODIF_VAR = 0x0010;
const VPKT_EOT = 0xdd00;
const VPKT_DELAY_ACK = 0xbb00;
const VPKT_ERROR = 0xee00;

const EXEC_PRGM = 0;
const EXEC_ASM = 1;
const EXEC_APP = 2;

const AID_ARCHIVED = 0x0003;
const AID_VAR_VERSION = 0x0008;
const AID_VAR_SIZE = 0x0001;
const AID_VAR_TYPE2 = 0x0011;
const AID_VAR_TYPE = 0x0002;

const VAR_TYPE_APPVAR = 0x15;
const VAR_TYPE_PROTECTED_PRGM = 0x06;

const PID_SCREENSHOT = 0x0022;

const USB_ERROR_CODES = [
  0x0004, 0x0006, 0x0008, 0x0009, 0x000c, 0x000d, 0x000e, 0x0011,
  0x0012, 0x001b, 0x001c, 0x001d, 0x0021, 0x0022, 0x0023, 0x0027,
  0x0029, 0x002b, 0x002c, 0x002d, 0x002e, 0x002f, 0x0030, 0x0034,
];

const USB_ERROR_MESSAGES = [
  "invalid argument or name",
  "cannot delete var/app from archive",
  "transmission error",
  "basic mode while in boot mode",
  "out of memory",
  "invalid name",
  "invalid name",
  "busy",
  "can't overwrite locked variable",
  "variable too large",
  "mode token too small",
  "mode token too large",
  "wrong size for parameter",
  "invalid parameter ID",
  "read-only parameter",
  "wrong modify request",
  "remote control error",
  "battery low",
  "flash application rejected",
  "flash application rejected",
  "flash application rejected: signature does not match",
  "flash application rejected",
  "flash application rejected",
  "hand-held is busy; go to HOME screen",
];

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    }),
  ]);
}

function bytesBE(value, width) {
  const out = new Array(width);
  for (let i = width - 1; i >= 0; i -= 1) {
    out[i] = value & 0xff;
    value = Math.floor(value / 256);
  }
  return out;
}

function intBE(bytes) {
  let value = 0;
  for (const byte of bytes) {
    value = value * 256 + byte;
  }
  return value;
}

function intLE(bytes) {
  let value = 0;
  for (let i = bytes.length - 1; i >= 0; i -= 1) {
    value = value * 256 + bytes[i];
  }
  return value;
}

function int32LE(bytes) {
  return (
    (bytes[0] || 0) |
    ((bytes[1] || 0) << 8) |
    ((bytes[2] || 0) << 16) |
    ((bytes[3] || 0) << 24)
  ) >>> 0;
}

function int24sLE(bytes) {
  const value = intLE(bytes);
  return value & 0x800000 ? value - 0x1000000 : value;
}

function asciiZ(bytes) {
  const buffer = Buffer.from(bytes);
  const zero = buffer.indexOf(0);
  return buffer.slice(0, zero >= 0 ? zero : buffer.length).toString("ascii");
}

module.exports = {
  VPKT_EXECUTE,
  VPKT_DATA_ACK,
  VPKT_PARM_REQ,
  VPKT_PARM_DATA,
  VPKT_DIR_REQ,
  VPKT_RTS,
  VPKT_VAR_HDR,
  VPKT_VAR_REQ,
  VPKT_VAR_CNTS,
  VPKT_MODIF_VAR,
  VPKT_EOT,
  VPKT_DELAY_ACK,
  VPKT_ERROR,
  EXEC_PRGM,
  EXEC_ASM,
  EXEC_APP,
  AID_ARCHIVED,
  AID_VAR_VERSION,
  AID_VAR_SIZE,
  AID_VAR_TYPE2,
  AID_VAR_TYPE,
  VAR_TYPE_APPVAR,
  VAR_TYPE_PROTECTED_PRGM,
  PID_SCREENSHOT,
  USB_ERROR_CODES,
  USB_ERROR_MESSAGES,
  withTimeout,
  bytesBE,
  intBE,
  intLE,
  int32LE,
  int24sLE,
  asciiZ,
};
