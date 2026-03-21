const express=require('express');
const router=express.Router();
const {handleTwilioRecording,handleIncomingCall,handleIncomingSms,pollAgentResponse}=require('../controllers/webhookController');

router.post('/voice', handleIncomingCall);
router.post('/twilio/webhook',handleTwilioRecording);
router.post('/sms/webhook', handleIncomingSms);
router.get('/sms/poll', pollAgentResponse);
router.post('/sms/poll', pollAgentResponse);
module.exports=router;