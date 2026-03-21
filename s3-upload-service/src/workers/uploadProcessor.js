const axios = require("axios");
const S3Client = require("../config/s3");
const { Upload } = require("@aws-sdk/lib-storage");
const { SQSClient, SendMessageCommand } = require("@aws-sdk/client-sqs");

const sqsClient = new SQSClient({
  region: process.env.AWS_REGION || "ap-south-1",
});

async function uploadProcessor(job) {
  const { jobType = "voice-recording", RecordingUrl, filename, callerNumber, callSid, smsText, messageSid } = job.data;
  console.log(`[Worker] Processing job ${job.id}: type=${jobType}`);

  if (jobType === "sms") {
    try {
      const queueUrl =
        process.env.AGENT_JOBS_QUEUE_URL ||
        "https://sqs.ap-south-1.amazonaws.com/963716652927/ai-agent-jobs";
      if (!queueUrl) {
        throw new Error("AGENT_JOBS_QUEUE_URL is not configured");
      }

      const payload = {
        transcription: smsText,
        "caller-number": callerNumber,
        source: "sms",
        message_id: messageSid || "unknown",
      };

      await sqsClient.send(
        new SendMessageCommand({
          QueueUrl: queueUrl,
          MessageBody: JSON.stringify(payload),
        })
      );

      console.log(`[Worker] Job ${job.id} completed: SMS forwarded to AGENT_JOBS_QUEUE_URL.`);
      return { success: true, forwarded: true };
    } catch (error) {
      console.error(`[Worker] Job ${job.id} failed (sms):`, error);
      throw error;
    }
  }

  console.log(`[Worker] Uploading ${filename}`);
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
