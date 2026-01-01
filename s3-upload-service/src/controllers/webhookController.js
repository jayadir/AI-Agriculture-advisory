const uploadQueue=require('../queues/uploadQueue');
const {VoiceResponse}=require('twilio').twiml;
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

module.exports={handleTwilioRecording, handleIncomingCall};