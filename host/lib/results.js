const { asciiZ, intBE, intLE, int24sLE } = require("./protocol");

function parseVarHeader(data) {
  let offset = 0;
  const folderLen = data[offset++];
  const folder = folderLen ? asciiZ(data.slice(offset, offset + folderLen + 1)) : "";
  offset += folderLen ? folderLen + 1 : 0;

  const nameLen = data[offset++];
  const name = nameLen ? asciiZ(data.slice(offset, offset + nameLen + 1)) : "";
  offset += nameLen ? nameLen + 1 : 0;

  const attrCount = intBE(data.slice(offset, offset + 2));
  offset += 2;

  const attrs = [];
  for (let i = 0; i < attrCount; i += 1) {
    const id = intBE(data.slice(offset, offset + 2));
    const ok = data[offset + 2] === 0;
    offset += 3;
    let raw = [];

    if (ok) {
      const size = intBE(data.slice(offset, offset + 2));
      offset += 2;
      raw = Array.from(data.slice(offset, offset + size));
      offset += size;
    }

    attrs.push({ id, ok, raw, value: raw.length ? intBE(raw) : undefined });
  }

  return { folder, name, attrs };
}

function parseBenchResult(data) {
  const inferMagic = Buffer.from("MKRN", "ascii");
  const genMagic = Buffer.from("GENR", "ascii");
  const magic = Buffer.from("RNNB", "ascii");
  let offset = -1;

  for (let i = 0; i <= data.length - inferMagic.length; i += 1) {
    if (Buffer.from(data.slice(i, i + inferMagic.length)).equals(inferMagic)) {
      let cursor = i + 4;
      const version = data[cursor++];
      const h = data[cursor++];
      const vocab = data[cursor++];
      const tokens = data[cursor++];
      const readU32 = () => {
        const value = intLE(data.slice(cursor, cursor + 4));
        cursor += 4;
        return value;
      };
      const readS24 = () => {
        const value = int24sLE(data.slice(cursor, cursor + 3));
        cursor += 3;
        return value;
      };
      const parsed = {
        magic_offset: i,
        kind: "megakernel",
        version,
        h,
        vocab,
        tokens,
      };

      if (data.length < cursor + 24) {
        return parsed;
      }

      parsed.cycles_recurrent_only = readU32();
      parsed.cycles_output_only = readU32();
      parsed.cycles_full_generate = readU32();
      parsed.recurrent_checksum = readS24();
      parsed.output_checksum = readS24();
      parsed.full_checksum = readS24();
      parsed.last_id = data[cursor++];
      parsed.generated = Buffer.from(data.slice(cursor, cursor + 65)).toString("ascii").replace(/\0.*$/s, "");
      cursor += 65;
      if (version >= 2 && data.length >= cursor + 87) {
        parsed.cycles_recurrent_fused = readU32();
        parsed.cycles_output_fused = readU32();
        parsed.cycles_full_generate_fused = readU32();
        parsed.recurrent_fused_checksum = readS24();
        parsed.output_fused_checksum = readS24();
        parsed.full_fused_checksum = readS24();
        parsed.fused_last_id = data[cursor++];
        parsed.generated_fused = Buffer.from(data.slice(cursor, cursor + 65)).toString("ascii").replace(/\0.*$/s, "");
        cursor += 65;
      }
      if (version >= 3 && data.length >= cursor + 80) {
        parsed.cycles_output_argmax = readU32();
        parsed.cycles_full_generate_argmax = readU32();
        parsed.output_argmax_checksum = readS24();
        parsed.full_argmax_checksum = readS24();
        parsed.argmax_last_id = data[cursor++];
        parsed.generated_argmax = Buffer.from(data.slice(cursor, cursor + 65)).toString("ascii").replace(/\0.*$/s, "");
      }
      return parsed;
    }
  }

  for (let i = 0; i <= data.length - genMagic.length; i += 1) {
    if (Buffer.from(data.slice(i, i + genMagic.length)).equals(genMagic)) {
      let cursor = i + 4;
      const readU16 = () => {
        const value = intLE(data.slice(cursor, cursor + 2));
        cursor += 2;
        return value;
      };
      const readU32 = () => {
        const value = intLE(data.slice(cursor, cursor + 4));
        cursor += 4;
        return value;
      };
      const version = data[cursor++];
      const h = data[cursor++];
      const vocab = version >= 2 ? readU16() : data[cursor++];
      const tokens = readU16();
      const cycles_generate = readU32();
      const last_id = version >= 2 ? readU16() : data[cursor++];
      return {
        magic_offset: i,
        kind: "generated_text",
        version,
        h,
        vocab,
        tokens,
        cycles_generate,
        last_id,
        text: Buffer.from(data.slice(cursor, cursor + tokens + 1)).toString("ascii").replace(/\0.*$/s, ""),
      };
    }
  }

  for (let i = 0; i <= data.length - magic.length; i += 1) {
    if (Buffer.from(data.slice(i, i + magic.length)).equals(magic)) {
      offset = i;
      break;
    }
  }

  if (offset < 0 || data.length < offset + 29) {
    return null;
  }

  let cursor = offset + 4;
  const version = data[cursor++];
  const h = data[cursor++];
  const iters = intLE(data.slice(cursor, cursor + 2));
  cursor += 2;

  const readU24 = () => {
    const value = intLE(data.slice(cursor, cursor + 3));
    cursor += 3;
    return value;
  };
  const readU32 = () => {
    const value = intLE(data.slice(cursor, cursor + 4));
    cursor += 4;
    return value;
  };
  const readS24 = () => {
    const value = int24sLE(data.slice(cursor, cursor + 3));
    cursor += 3;
    return value;
  };

  const parsed = {
    magic_offset: offset,
    version,
    h,
    iters,
  };

  if (version >= 3) {
    if (data.length < cursor + 48) {
      return parsed;
    }
    parsed.cycles_c_dot64 = readU32();
    parsed.cycles_asm_dot64 = readU32();
    parsed.cycles_gru64_c_step = readU32();
    parsed.cycles_gru64_asm_step = readU32();
    parsed.cycles_lut256 = readU32();
    parsed.dot_delta = readS24();
    parsed.checksum = readS24();
    parsed.cycles_matvec_c64 = readU32();
    parsed.cycles_matvec_dotcall64 = readU32();
    parsed.cycles_matvec_asm64 = readU32();
    parsed.cycles_matvec3_asm64 = readU32();
    parsed.matvec_delta = readS24();
    parsed.matvec_checksum = readS24();

    if (version >= 4 && data.length >= cursor + 18) {
      parsed.cycles_matvec_offset_c64 = readU32();
      parsed.cycles_matvec_offset_asm64 = readU32();
      parsed.cycles_matvec3_offset_asm64 = readU32();
      parsed.offset_delta = readS24();
      parsed.offset_checksum = readS24();
    }

    if (version >= 5 && data.length >= cursor + 18) {
      parsed.cycles_matvec_offset_store64 = readU32();
      parsed.cycles_matvec3_offset_store64 = readU32();
      parsed.cycles_gru64_offset_step = readU32();
      parsed.store_delta = readS24();
      parsed.gru_offset_checksum = readS24();
    }

    if (version >= 10 && data.length >= cursor + 7) {
      parsed.cycles_rnn64_offset_step = readU32();
      parsed.rnn_offset_checksum = readS24();
    }

    return parsed;
  }

  parsed.cycles_c_dot64 = readU24();
  parsed.cycles_asm_dot64 = readU24();
  parsed.cycles_gru64_c_step = readU24();
  parsed.cycles_gru64_asm_step = readU24();
  parsed.cycles_lut256 = readU24();
  parsed.dot_delta = readS24();
  parsed.checksum = readS24();

  if (version >= 2 && data.length >= cursor + 18) {
    parsed.cycles_matvec_c64 = readU24();
    parsed.cycles_matvec_dotcall64 = readU24();
    parsed.cycles_matvec_asm64 = readU24();
    parsed.cycles_matvec3_asm64 = readU24();
    parsed.matvec_delta = readS24();
    parsed.matvec_checksum = readS24();
  }

  return parsed;
}

function parseParameters(data) {
  const count = intBE(data.slice(0, 2));
  let offset = 2;
  const params = [];

  for (let i = 0; i < count && offset + 5 <= data.length; i += 1) {
    const type = intBE(data.slice(offset, offset + 2));
    const ok = data[offset + 2] === 0;
    const size = intBE(data.slice(offset + 3, offset + 5));
    const raw = Array.from(data.slice(offset + 5, offset + 5 + size));
    params.push({ type, ok, size, raw, value: raw.length <= 4 ? intBE(raw) : undefined });
    offset += 5 + size;
  }

  return params;
}

module.exports = {
  parseVarHeader,
  parseBenchResult,
  parseParameters,
};
