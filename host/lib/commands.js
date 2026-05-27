const fs = require("fs");
const path = require("path");

const {
  VPKT_EXECUTE,
  VPKT_DATA_ACK,
  VPKT_PARM_REQ,
  VPKT_PARM_DATA,
  VPKT_DIR_REQ,
  VPKT_VAR_HDR,
  VPKT_VAR_REQ,
  VPKT_VAR_CNTS,
  VPKT_MODIF_VAR,
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
  bytesBE,
  intBE,
  withTimeout,
} = require("./protocol");
const { commandSend, commandSendApp } = require("./files");
const { recvAnyVirtual, recvVirtualPacket } = require("./packets");
const { parseVarHeader, parseBenchResult, parseParameters } = require("./results");

const USAGE = "usage: node host/ti84-webusb.js info|send <file.8xp>|sendapp <file.8ek>|sendram <file.8xp>|sendram5 <file.8xp>|exec <program>|execasm <program>|execapp <app>|appvar <name>|archive <program>|unarchive <program>|delete <program>|deleteappvar <name>|dir|param <id>|screenshot [file.ppm]|key <byte>";

async function requestParameters(calc, ids) {
  const payload = [
    ...bytesBE(ids.length, 2),
    ...ids.flatMap((id) => bytesBE(id, 2)),
  ];
  await withTimeout(calc._d.send({ type: VPKT_PARM_REQ, data: payload }), 5000, "param request");
  const packet = await withTimeout(calc._d.expect(VPKT_PARM_DATA), 5000, "param response");
  return parseParameters(packet.data);
}

async function commandInfo(calc) {
  const ready = await withTimeout(calc.isReady(), 5000, "isReady");
  console.log(`ready=${ready}`);
  if (!ready) {
    return;
  }

  const params = await requestParameters(calc, [0x000e, 0x0011]);
  const ram = params.find((param) => param.type === 0x000e);
  const flash = params.find((param) => param.type === 0x0011);
  console.log(`model=${calc.name}`);
  console.log(`free_ram=${ram && ram.ok ? intBE(ram.raw) : "unknown"}`);
  console.log(`free_flash=${flash && flash.ok ? intBE(flash.raw) : "unknown"}`);
}

async function commandKey(calc, key) {
  const value = Number.parseInt(key, key.startsWith("0x") ? 16 : 10);
  if (!Number.isInteger(value) || value < 0 || value > 255) {
    throw new Error("key must be a byte, for example 0x05");
  }

  await withTimeout(calc.pressKey(value), 5000, "pressKey");
  console.log(`pressed=0x${value.toString(16).padStart(2, "0")}`);
}

async function commandExec(calc, name, executeType = EXEC_PRGM) {
  if (!/^[A-Za-z0-9_]{1,8}$/.test(name)) {
    throw new Error("program name must be 1-8 ASCII letters/digits/underscore");
  }

  const bytes = Buffer.from(name.toUpperCase(), "ascii");
  const payload = [
    0,
    bytes.length,
    ...bytes,
    0,
    executeType,
  ];

  await withTimeout(calc._d.send({ type: VPKT_EXECUTE, data: payload }), 5000, "execute send");
  await withTimeout(calc._d.expect(VPKT_DATA_ACK), 5000, "execute ack");
  const label = executeType === EXEC_ASM ? "executed_asm" : executeType === EXEC_APP ? "executed_app" : "executed";
  console.log(`${label}=${name.toUpperCase()}`);
}

function buildVarRequest(name) {
  const bytes = Buffer.from(name.toUpperCase(), "ascii");
  const aids = [AID_ARCHIVED, AID_VAR_VERSION, AID_VAR_SIZE];
  const attrs = [
    {
      id: AID_VAR_TYPE2,
      raw: [0xf0, 0x07, 0x00, VAR_TYPE_APPVAR],
    },
  ];

  return [
    0,
    bytes.length,
    ...bytes,
    0,
    0x01,
    0xff,
    0xff,
    0xff,
    0xff,
    ...bytesBE(aids.length, 2),
    ...aids.flatMap((aid) => bytesBE(aid, 2)),
    ...bytesBE(attrs.length, 2),
    ...attrs.flatMap((attr) => [
      ...bytesBE(attr.id, 2),
      ...bytesBE(attr.raw.length, 2),
      ...attr.raw,
    ]),
    0,
    0,
  ];
}

async function commandDir(calc) {
  const aids = [AID_VAR_SIZE, AID_VAR_TYPE, AID_ARCHIVED];
  const payload = [
    0,
    0,
    ...bytesBE(aids.length, 2),
    ...aids.flatMap((aid) => bytesBE(aid, 2)),
    0,
    1,
    0,
    1,
    0,
    1,
    1,
  ];

  await withTimeout(calc._d.send({ type: VPKT_DIR_REQ, data: payload }), 5000, "dir request");

  for (let i = 0; i < 512; i += 1) {
    const packet = await withTimeout(recvAnyVirtual(calc), 10000, "dir packet");
    if (packet.type === 0xdd00) {
      console.log("dir_eot=true");
      return;
    }
    if (packet.type !== VPKT_VAR_HDR) {
      console.log(`dir_unexpected=0x${packet.type.toString(16)}`);
      continue;
    }

    const header = parseVarHeader(packet.data);
    const attrs = Object.fromEntries(header.attrs.map((attr) => [`0x${attr.id.toString(16)}`, attr]));
    const size = attrs["0x1"] && attrs["0x1"].value;
    const type = attrs["0x2"] && attrs["0x2"].raw ? Buffer.from(attrs["0x2"].raw).toString("hex") : "";
    const archived = attrs["0x3"] && attrs["0x3"].value;
    console.log(`var=${header.name} size=${size ?? ""} type=${type} archived=${archived ?? ""}`);
  }
}

async function commandParam(calc, idText) {
  const id = Number.parseInt(idText, idText.startsWith("0x") ? 16 : 10);
  if (!Number.isInteger(id) || id < 0 || id > 0xffff) {
    throw new Error("param id must be a 16-bit integer, for example 0x37");
  }

  await withTimeout(calc._d.send({ type: VPKT_PARM_REQ, data: [0, 1, ...bytesBE(id, 2)] }), 5000, "param request");
  const packet = await withTimeout(calc._d.expect(VPKT_PARM_DATA), 5000, "param response");
  const params = parseParameters(packet.data);
  for (const param of params) {
    console.log(`param=0x${param.type.toString(16)} ok=${param.ok} size=${param.size}`);
    console.log(`hex=${Buffer.from(param.raw).toString("hex")}`);
    const printable = Buffer.from(param.raw).toString("latin1").replace(/[^\x20-\x7e\r\n]/g, ".");
    console.log(`ascii=${printable}`);
  }
}

async function commandScreenshot(calc, outPath = "tmp/screenshot.ppm") {
  await withTimeout(
    calc._d.send({ type: VPKT_PARM_REQ, data: [0, 1, ...bytesBE(PID_SCREENSHOT, 2)] }),
    5000,
    "screenshot request",
  );
  const packet = await withTimeout(recvVirtualPacket(calc), 60000, "screenshot response");
  if (packet.type !== VPKT_PARM_DATA) {
    throw new Error(`expected screenshot parameter data, got 0x${packet.type.toString(16)}`);
  }

  const params = parseParameters(packet.data);
  const shot = params.find((param) => param.type === PID_SCREENSHOT && param.ok);
  if (!shot) {
    throw new Error("calculator did not return screenshot data");
  }
  if (shot.raw.length < 320 * 240 * 2) {
    throw new Error(`unexpected screenshot size ${shot.raw.length}`);
  }

  const rgb = Buffer.alloc(320 * 240 * 3);
  for (let i = 0, j = 0; i < 320 * 240; i += 1, j += 2) {
    const word = (shot.raw[j] << 8) | shot.raw[j + 1];
    const r = (word >> 11) & 0x1f;
    const g = (word >> 5) & 0x3f;
    const b = word & 0x1f;
    rgb[i * 3 + 0] = (r << 3) | (r >> 2);
    rgb[i * 3 + 1] = (g << 2) | (g >> 4);
    rgb[i * 3 + 2] = (b << 3) | (b >> 2);
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, Buffer.concat([Buffer.from("P6\n320 240\n255\n", "ascii"), rgb]));
  console.log(`screenshot=${outPath}`);
  console.log(`bytes=${shot.raw.length}`);
}

async function commandAppVar(calc, name) {
  if (!/^[A-Za-z0-9_]{1,8}$/.test(name)) {
    throw new Error("appvar name must be 1-8 ASCII letters/digits/underscore");
  }

  await withTimeout(calc._d.send({ type: VPKT_VAR_REQ, data: buildVarRequest(name) }), 5000, "var request send");
  const headerPacket = await withTimeout(recvVirtualPacket(calc), 15000, "var header");
  if (headerPacket.type !== VPKT_VAR_HDR) {
    throw new Error(`expected AppVar header, got 0x${headerPacket.type.toString(16)}`);
  }

  const contentPacket = await withTimeout(recvVirtualPacket(calc), 30000, "var content");
  if (contentPacket.type !== VPKT_VAR_CNTS) {
    throw new Error(`expected AppVar content, got 0x${contentPacket.type.toString(16)}`);
  }

  const header = parseVarHeader(headerPacket.data);
  const content = Array.from(contentPacket.data);
  const parsed = parseBenchResult(content);

  console.log(`appvar=${header.name || name.toUpperCase()}`);
  console.log(`bytes=${content.length}`);
  console.log(`attrs=${JSON.stringify(header.attrs)}`);
  console.log(`hex=${Buffer.from(content).toString("hex")}`);
  if (parsed) {
    for (const [key, value] of Object.entries(parsed)) {
      console.log(`${key}=${value}`);
    }
  }
}

function buildAttrChange(name, type, archived) {
  const bytes = Buffer.from(name.toUpperCase(), "ascii");
  return [
    0,
    bytes.length,
    ...bytes,
    0,
    0,
    1,
    ...bytesBE(AID_VAR_TYPE2, 2),
    0,
    4,
    0xf0,
    0x07,
    0x00,
    type,
    0x01,
    0,
    bytes.length,
    ...bytes,
    0,
    0,
    1,
    ...bytesBE(AID_ARCHIVED, 2),
    0,
    1,
    archived ? 0x01 : 0x00,
  ];
}

function buildDelete(name, type) {
  const bytes = Buffer.from(name.toUpperCase(), "ascii");
  return [
    0,
    bytes.length,
    ...bytes,
    0,
    0,
    1,
    ...bytesBE(AID_VAR_TYPE2, 2),
    0,
    4,
    0xf0,
    0x07,
    0x00,
    type,
    0x01,
    0,
    0,
    0,
    0,
  ];
}

async function commandArchive(calc, name, archived) {
  if (!/^[A-Za-z0-9_]{1,8}$/.test(name)) {
    throw new Error("program name must be 1-8 ASCII letters/digits/underscore");
  }

  await withTimeout(
    calc._d.send({ type: VPKT_MODIF_VAR, data: buildAttrChange(name, VAR_TYPE_PROTECTED_PRGM, archived) }),
    5000,
    "attribute change send",
  );
  await withTimeout(calc._d.expect(VPKT_DATA_ACK), 5000, "attribute change ack");
  console.log(`${archived ? "archived" : "unarchived"}=${name.toUpperCase()}`);
}

async function commandDelete(calc, name, type = VAR_TYPE_PROTECTED_PRGM) {
  if (!/^[A-Za-z0-9_]{1,8}$/.test(name)) {
    throw new Error("program name must be 1-8 ASCII letters/digits/underscore");
  }

  await withTimeout(
    calc._d.send({ type: VPKT_MODIF_VAR, data: buildDelete(name, type) }),
    5000,
    "delete send",
  );
  await withTimeout(calc._d.expect(VPKT_DATA_ACK), 5000, "delete ack");
  console.log(`deleted=${name.toUpperCase()}`);
}

async function runCommand(calc, command, arg) {
  if (command === "info") {
    await commandInfo(calc);
  } else if (command === "send") {
    if (!arg) {
      throw new Error("send requires a .8xp path");
    }
    await commandSend(calc, arg);
  } else if (command === "sendapp") {
    if (!arg) {
      throw new Error("sendapp requires a .8ek path");
    }
    await commandSendApp(calc, arg);
  } else if (command === "sendram") {
    if (!arg) {
      throw new Error("sendram requires a .8xp path");
    }
    await commandSend(calc, arg, { archived: false });
  } else if (command === "sendram5") {
    if (!arg) {
      throw new Error("sendram5 requires a .8xp path");
    }
    await commandSend(calc, arg, { archived: false, type: 5 });
  } else if (command === "key") {
    if (!arg) {
      throw new Error("key requires a key byte");
    }
    await commandKey(calc, arg);
  } else if (command === "exec") {
    if (!arg) {
      throw new Error("exec requires a program name");
    }
    await commandExec(calc, arg);
  } else if (command === "execasm") {
    if (!arg) {
      throw new Error("execasm requires a program name");
    }
    await commandExec(calc, arg, EXEC_ASM);
  } else if (command === "execapp") {
    if (!arg) {
      throw new Error("execapp requires an app name");
    }
    await commandExec(calc, arg, EXEC_APP);
  } else if (command === "appvar") {
    if (!arg) {
      throw new Error("appvar requires an appvar name");
    }
    await commandAppVar(calc, arg);
  } else if (command === "param") {
    if (!arg) {
      throw new Error("param requires an id");
    }
    await commandParam(calc, arg);
  } else if (command === "dir") {
    await commandDir(calc);
  } else if (command === "archive" || command === "unarchive") {
    if (!arg) {
      throw new Error(`${command} requires a program name`);
    }
    await commandArchive(calc, arg, command === "archive");
  } else if (command === "delete") {
    if (!arg) {
      throw new Error("delete requires a program name");
    }
    await commandDelete(calc, arg);
  } else if (command === "deleteappvar") {
    if (!arg) {
      throw new Error("deleteappvar requires an appvar name");
    }
    await commandDelete(calc, arg, VAR_TYPE_APPVAR);
  } else if (command === "screenshot") {
    await commandScreenshot(calc, arg || "tmp/screenshot.ppm");
  } else {
    throw new Error(`unknown command: ${command}`);
  }
}

module.exports = {
  USAGE,
  runCommand,
};
