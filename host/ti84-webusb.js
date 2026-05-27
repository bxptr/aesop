#!/usr/bin/env node

const { connect, close } = require("./lib/connection");
const { USAGE, runCommand } = require("./lib/commands");

async function main(argv = process.argv.slice(2)) {
  const [command, arg] = argv;
  if (!command || command === "help") {
    console.log(USAGE);
    return;
  }

  const calc = await connect();
  try {
    await runCommand(calc, command, arg);
  } finally {
    await close(calc);
  }
}

main().then(() => process.exit(0)).catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
