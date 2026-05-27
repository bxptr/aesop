const fs = require("fs");
const path = require("path");
const { tifiles } = require("ticalc-usb");

const {
  VPKT_DATA_ACK,
  VPKT_RTS,
  VPKT_VAR_CNTS,
  VPKT_EOT,
  AID_ARCHIVED,
  AID_VAR_TYPE,
  bytesBE,
  int32LE,
  asciiZ,
  withTimeout,
} = require("./protocol");
const { expectVirtual } = require("./packets");

async function commandSend(calc, filePath, overrides = {}) {
  const bytes = new Uint8Array(fs.readFileSync(filePath));
  const parsed = tifiles.parseFile(bytes);

  if (!tifiles.isValid(parsed)) {
    throw new Error(`${filePath} is not a valid TI file`);
  }
  for (const entry of parsed.entries) {
    if (overrides.archived !== undefined) {
      entry.attributes = entry.attributes || {};
      entry.attributes.archived = overrides.archived;
    }
    if (overrides.type !== undefined) {
      entry.type = overrides.type;
    }
  }
  if (!calc.canReceive(parsed)) {
    throw new Error(`${calc.name} cannot receive ${filePath}`);
  }

  try {
    const details = await withTimeout(calc.getStorageDetails(parsed), 5000, "getStorageDetails");
    if (!details.fits) {
      throw new Error(
        `Not enough calculator storage: free=${JSON.stringify(details.free)}, required=${JSON.stringify(details.required)}`,
      );
    }
  } catch (error) {
    console.warn(`storage_check=skipped (${error})`);
  }

  const sendTimeout = Math.min(300000, Math.max(15000, Math.ceil(bytes.length / 4096) * 2000));
  await withTimeout(calc.sendFile(parsed), sendTimeout, "sendFile");
  console.log(`sent=${path.basename(filePath)}`);
  for (const entry of parsed.entries) {
    console.log(`entry=${entry.name} type=${entry.type} archived=${entry.attributes && entry.attributes.archived}`);
  }
}

function buildSendParameters(attrs) {
  return [
    ...bytesBE(attrs.length, 2),
    ...attrs.flatMap((attr) => [
      ...bytesBE(attr.id, 2),
      ...bytesBE(attr.raw.length, 2),
      ...attr.raw,
    ]),
  ];
}

function parseTiflApp(bytes, filePath) {
  if (bytes.length < 78 || Buffer.from(bytes.slice(0, 8)).toString("ascii") !== "**TIFL**") {
    throw new Error(`${filePath} is not a TI flash app file`);
  }

  const name = asciiZ(bytes.slice(0x11, 0x19));
  const deviceType = bytes[0x30];
  const dataType = bytes[0x31];
  const hwId = bytes[0x49];
  const dataLength = int32LE(bytes.slice(0x4a, 0x4e));
  const dataStart = 0x4e;
  const dataEnd = dataStart + dataLength;
  if (!/^[A-Za-z0-9_]{1,8}$/.test(name)) {
    throw new Error(`${filePath} has unsupported app name ${JSON.stringify(name)}`);
  }
  if (dataEnd > bytes.length) {
    throw new Error(`${filePath} is truncated: header length=${dataLength}, file bytes=${bytes.length}`);
  }
  if (dataType !== 0x24 || hwId === 0 || bytes[dataStart] !== 0x81) {
    throw new Error(
      `${filePath} is not a native CE flash app payload (device=0x${deviceType.toString(16)}, ` +
      `type=0x${dataType.toString(16)}, hw=0x${hwId.toString(16)})`,
    );
  }

  return {
    name,
    type: dataType,
    size: dataLength,
    data: bytes.slice(dataStart, dataEnd),
  };
}

async function commandSendApp(calc, filePath) {
  const bytes = new Uint8Array(fs.readFileSync(filePath));
  const app = parseTiflApp(bytes, filePath);
  const attrs = buildSendParameters([
    {
      id: AID_VAR_TYPE,
      raw: [0xf0, 0x0f, 0x00, app.type],
    },
    {
      id: AID_ARCHIVED,
      raw: [0x01],
    },
  ]);
  const payload = [
    ...bytesBE(app.name.length, 2),
    ...Buffer.from(app.name, "ascii"),
    0,
    ...bytesBE(app.size, 4),
    1,
    ...attrs,
  ];

  await withTimeout(calc._d.send({ type: VPKT_RTS, data: payload }), 5000, "app rts send");
  await expectVirtual(calc, VPKT_DATA_ACK, "app rts ack");
  await withTimeout(calc._d.send({ type: VPKT_VAR_CNTS, data: Array.from(app.data) }), 300000, "app content send");
  await expectVirtual(calc, VPKT_DATA_ACK, "app content ack");
  await withTimeout(calc._d.send({ type: VPKT_EOT, data: [] }), 5000, "app eot send");

  console.log(`sent_app=${path.basename(filePath)}`);
  console.log(`entry=${app.name} type=${app.type} bytes=${app.size}`);
}

module.exports = {
  commandSend,
  commandSendApp,
};
