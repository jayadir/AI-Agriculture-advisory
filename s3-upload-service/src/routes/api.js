const express=require('express');
const router=express.Router();
const {handleTwilioRecording,handleIncomingCall}=require('../controllers/webhookController');

router.post('/voice', handleIncomingCall);
router.post('/twilio/webhook',handleTwilioRecording);
module.exports=router;