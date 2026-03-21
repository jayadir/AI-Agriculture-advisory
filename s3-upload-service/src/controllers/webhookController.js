const uploadQueue=require('../queues/uploadQueue');
const {VoiceResponse}=require('twilio').twiml;
const { SQSClient, ReceiveMessageCommand, DeleteMessageCommand, GetQueueUrlCommand } = require('@aws-sdk/client-sqs');

const sqsClient = new SQSClient({ region: process.env.AWS_REGION || 'ap-south-1' });
const AGENT_RESPONSE_QUEUE_URL = process.env.AGENT_RESPONSE_QUEUE_URL;
const AGENT_RESPONSE_QUEUE_NAME = process.env.AGENT_RESPONSE_QUEUE_NAME || 'agent-response-queue';

function isLikelyValidSqsQueueUrl(queueUrl) {
    if (!queueUrl || typeof queueUrl !== 'string') {
        return false;
    }

    const trimmed = queueUrl.trim();
    if (!trimmed.startsWith('https://')) {
        return false;
    }

    const withoutProtocol = trimmed.replace('https://', '');
    const slashCount = (withoutProtocol.match(/\//g) || []).length;
    return slashCount >= 2;
}

async function resolveAgentResponseQueueUrl() {
    if (isLikelyValidSqsQueueUrl(AGENT_RESPONSE_QUEUE_URL)) {
        return AGENT_RESPONSE_QUEUE_URL.trim();
    }

    const resolved = await sqsClient.send(
        new GetQueueUrlCommand({
            QueueName: AGENT_RESPONSE_QUEUE_NAME,
        })
    );

    if (!resolved.QueueUrl) {
        throw new Error('Unable to resolve SQS queue URL for agent responses.');
    }

    return resolved.QueueUrl;
}

const handleTwilioRecording=async (req,res)=>{
    try {
        const {RecordingUrl,RecordingSid,From}=req.body;
        if(!RecordingUrl||!RecordingSid){
            console.warn('Missing RecordingUrl or RecordingSid in webhook payload');
            return res.status(400).send('Bad Request: Missing parameters');
        }
        console.log(`[WebHook] Recording from: ${From}`);
        const filename=`recordings/${RecordingSid}.wav`;
        await uploadQueue.add({
            jobType: "voice-recording",
            RecordingUrl,
            filename,
            callerNumber: From,
            callSid: req.body.CallSid
        },{
            attempts:3,
            backoff: 5000,    
            removeOnComplete: true
        });
    
        console.log(`Enqueued upload job for RecordingSid: ${RecordingSid}`);
        res.status(200).send('<Response></Response>');
    } catch (error) {
        console.error('Error handling Twilio webhook:', error);
        res.status(500).send('Internal Server Error');
    }
}

const handleIncomingCall = (req,res)=>{
    const twiml =new VoiceResponse();
    twiml.say('Hello, please leave a message after the beep.');
    twiml.record({
        action:"/api/twilio/webhook",
        method:"POST",
        maxLength:60,
        playBeep:true
    });
    twiml.hangup(); 

    res.type('text/xml');
    res.send(twiml.toString());
}

const handleIncomingSms = async (req, res) => {
    try {
        const from = req.body.From || req.body.sender || req.body.phone;
        const body = req.body.Body || req.body.text || req.body.message;
        const messageSid = req.body.MessageSid || req.body.message_id || "unknown";

        if (!from || !body) {
            console.warn("Missing From or Body in SMS payload");
            return res.status(400).send("Bad Request: Missing SMS fields");
        }

        await uploadQueue.add(
            {
                jobType: "sms",
                callerNumber: from,
                smsText: body,
                messageSid,
            },
            {
                attempts: 3,
                backoff: 5000,
                removeOnComplete: true,
            }
        );

        console.log(`[WebHook] Enqueued SMS job from ${from}`);
        return res.status(200).send("OK");
    } catch (error) {
        console.error("Error handling SMS webhook:", error);
        return res.status(500).send("Internal Server Error");
    }
}

const pollAgentResponse = async (req, res) => {
    try {
        const queueUrl = await resolveAgentResponseQueueUrl();

        const command = new ReceiveMessageCommand({
            QueueUrl: queueUrl,
            MaxNumberOfMessages: 1,
            WaitTimeSeconds: 2,
        });
        
        const response = await sqsClient.send(command);
        
        if (response.Messages && response.Messages.length > 0) {
            const message = response.Messages[0];
            const body = JSON.parse(message.Body);
            
            const deleteCommand = new DeleteMessageCommand({
                QueueUrl: queueUrl,
                ReceiptHandle: message.ReceiptHandle,
            });
            await sqsClient.send(deleteCommand);
            console.log(body)
            return res.status(200).json({
                success: true,
                message: "Response retrieved successfully",
                data: {
                    responseText: body.response,
                    callerNumber: body["phone"],
                }
            });
        } else {
            return res.status(404).json({
                success: false,
                message: "No responses available",
            });
        }
    } catch (error) {
        console.error("Error polling SQS:", error);
        return res.status(500).json({
            success: false,
            message: "Failed to poll response queue",
            error: error.message,
        });
    }
}

module.exports={handleTwilioRecording, handleIncomingCall, handleIncomingSms, pollAgentResponse};