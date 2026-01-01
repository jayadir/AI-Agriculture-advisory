require("dotenv").config();
const uploadQueue = require("./src/queues/uploadQueue");
const uploadProcessor = require("./src/workers/uploadProcessor");

console.log("Worker started, waiting for jobs...");
uploadQueue.process(uploadProcessor);