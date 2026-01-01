const Queue=require('bull');
const redisConfig=require('../config/redis');

const uploadQueue=new Queue("audio-uploads",{
    redis:redisConfig       
})
module.exports=uploadQueue;