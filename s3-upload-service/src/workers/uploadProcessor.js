const axios = require("axios");
const S3Client = require("../config/s3");
const { Upload } = require("@aws-sdk/lib-storage");
const uploadQueue = require("../queues/uploadQueue");

async function uploadProcessor(job) {
  const { RecordingUrl, filename, callerNumber, callSid } = job.data;
  console.log(`[Worker] Processing job ${job.id}: Uploading ${filename}`);
  try {
    const res = await axios.get(RecordingUrl, {
      responseType: "stream",
      auth: {
        username: process.env.TWILIO_API_KEY_SID,
        password: process.env.TWILIO_API_KEY_SECRET,
      },
    });

    const parallelUpload = new Upload({
      client: S3Client,
      params: {
        Bucket: process.env.S3_BUCKET_NAME,
        Key: filename,
        Body: res.data,
        ContentType: res.headers["content-type"],
        Metadata: {
          "caller-number": callerNumber,
          "call-sid": callSid,
          "upload-date": new Date().toISOString(),
        },
      },
    });
    await parallelUpload.done();
    console.log(
      `[Worker] Job ${job.id} completed: Uploaded ${filename} to S3.`
    );
    return { success: true, key: filename };
  } catch (error) {
    console.error(`[Worker] Job ${job.id} failed:`, error);
    throw error;
  }
}

module.exports = uploadProcessor;
