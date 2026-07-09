# Demo Execution Guide

Step-by-step guide for presenting the PyTorch Quantization Demo at PyTorch Conference 2026.

## Pre-Demo Checklist (Day Before)

### Technical Setup

- [ ] Reserve H200 GPUs (October 19-21, 2026)
- [ ] Deploy all 4 model variants to OpenShift AI
- [ ] Test model endpoints responding correctly
- [ ] Deploy demo application to OpenShift
- [ ] Get public URL for QR code generation
- [ ] Test audience interface on mobile devices
- [ ] Test presenter dashboard on demo laptop
- [ ] Record backup demo video
- [ ] Prepare simulation mode with realistic data
- [ ] Test failover to simulation mode

### Content Preparation

- [ ] Prepare opening remarks
- [ ] Prepare closing remarks with vLLM meetup details
- [ ] Prepare example prompts for quality comparison
- [ ] Create QR code with public URL
- [ ] Print QR code backup (poster size)

### Equipment

- [ ] Laptop fully charged
- [ ] HDMI adapter/cable
- [ ] Backup laptop with full setup
- [ ] Phone hotspot configured
- [ ] Wireless presenter/clicker
- [ ] Backup USB drive with all files

## Day of Demo Checklist (Morning)

### 2 Hours Before

- [ ] Arrive at venue
- [ ] Test venue WiFi connection
- [ ] Verify screen/projector working
- [ ] Test audio/microphone
- [ ] Load presenter dashboard
- [ ] Verify all 4 model endpoints responding
- [ ] Test QR code on venue network
- [ ] Confirm simulation mode toggle working
- [ ] Test WebSocket connections
- [ ] Reset demo metrics to zero

### 30 Minutes Before

- [ ] Final endpoint health checks
- [ ] Open presenter dashboard in full screen
- [ ] Display QR code on separate device
- [ ] Test send request from your phone
- [ ] Verify metrics updating in real-time
- [ ] Have backup materials ready
- [ ] Silence phone notifications
- [ ] Close unnecessary applications

## Demo Execution (10 Minutes)

### [0:00-1:30] Opening & Setup

**Script:**
> "Good morning! Who here has tried to run Llama-70B in production and seen the GPU bill?"
>
> *[Wait for reactions/hands]*
>
> "Today we're going to stress test the same PyTorch model in four different quantization levels—live, right now, with your help. FP32 full precision, FP16 half precision, INT8 quantized, and INT4 extreme quantization."

**Actions:**
- Show OpenShift AI console with all 4 models deployed
- Point out current metrics: all idle, low GPU usage
- Briefly explain what quantization is

**Key Point:** "We'll see which is fastest, which uses least memory, and where quality starts to break down."

### [1:30-2:00] Call to Action

**Script:**
> "Here's how this works. Scan this QR code on your phone. Pick a model variant—try multiple if you want—and when I say go, start tapping that big red button as fast as you can."

**Actions:**
- Display QR code prominently
- Show audience interface on screen briefly
- Wait for ~50+ people to connect
- Show participant count climbing

**Monitoring:**
- Watch participant count in presenter dashboard
- Aim for 50+ before starting
- Show current baseline metrics

### [2:00-5:30] The Stress Test (Main Event)

**Script:**
> "Alright, I'm seeing [X] people connected. Let's do this. Ready... everyone START TAPPING!"

**Narration Points (as metrics change):**

1. **GPU Memory:**
   > "Look at GPU memory usage—FP32 is using 80GB, INT4 only using 10GB. That's 8x difference!"

2. **Throughput:**
   > "INT4 is handling way more requests per second. Look at that—it's processing 3x more than FP32!"

3. **Latency:**
   > "FP16 is the sweet spot here. Half the latency of FP32, but maintaining quality."

4. **Cost:**
   > "Cost per request on FP32 is 4x higher than INT4. In production, that adds up fast."

5. **Breaking Point:**
   > "FP32 is hitting memory limits first... queue depth building up... it's struggling."

**Actions:**
- Point to specific metrics as they change
- Call out interesting patterns
- Let it run for 2-3 minutes of chaos
- Keep energy high

### [5:30-7:00] Quality Check & Analysis

**Script:**
> "Okay everyone, STOP! Let's see what happened."

**Actions:**
- Show pre-prepared response comparison
- Same complex prompt sent to all 4 models
- Display responses side-by-side

**Script:**
> "Here's the same reasoning task sent to each model. FP16 and INT8 responses are nearly identical to FP32. But look at INT4—you can see quality degradation on this complex reasoning task."
>
> "For simple prompts, INT4 is fine. For complex reasoning, you need FP16 or INT8."

**Key Messages:**
- FP16 and INT8 are production sweet spots
- INT4 good for simple tasks, struggles with complex reasoning
- FP32 is baseline but expensive and slow

### [7:00-9:00] Platform Story

**Script:**
> "Let's look at how OpenShift AI makes this possible."

**Actions:**
- Navigate through OpenShift AI console
- Show multi-model serving configuration
- Point out vLLM as serving backend
- Show GPU resource allocation
- Display cost attribution dashboard

**Key Points:**
1. "All 4 variants running simultaneously on OpenShift AI"
2. "vLLM—a PyTorch Foundation project—powers this efficient serving"
3. "Platform handles GPU scheduling and routing automatically"
4. "Cost attribution tells you exactly what each team is spending"

**Script:**
> "This is how you optimize PyTorch inference in production. Platform handles the infrastructure complexity, you focus on choosing the right quantization for your use case."

### [9:00-10:00] Closing & Call to Action

**Script:**
> "So what's the winner? For most production use cases: FP16 or INT8. Fast enough, cheap enough, maintains quality."

**Display final metrics summary**

**Saša Connection:**
> "Want to go deeper on vLLM optimization and inference techniques? Saša is running a vLLM and llm-d meetup at [TIME] in [LOCATION]. Come with your questions!"

**Booth Invitation:**
> "Visit our booth to try deploying your own quantized models on OpenShift AI. We've got trial accounts ready to go."

**Final Message:**
> "Thank you! Questions? Catch me at the booth or the vLLM meetup."

## Backup Plans

### If Audience is Small (<30 people)

- Activate simulation mode (Ctrl+Shift+S)
- Add synthetic background traffic
- Focus more on quality comparison than pure stress test

### If Network Fails

- Switch to simulation mode immediately
- Continue narration as if live
- Use pre-recorded metrics if needed

### If Models Don't Respond

- Switch to simulation mode
- Show pre-recorded successful run
- Pivot to platform configuration walkthrough

### If Complete Technical Failure

- Show backup video
- Narrate over it
- Turn into Q&A about quantization strategies

## Presenter Tips

### Energy & Pacing

- Maintain high energy during stress test
- Speak clearly (expo floor is noisy)
- Use presenter clicker to point at metrics
- Keep moving to maintain audience engagement

### Audience Engagement

- Make eye contact
- React to metrics changes enthusiastically
- Acknowledge audience participation
- Thank them for helping stress test

### Time Management

- Glance at watch/timer occasionally
- Have a "short version" ready if running long
- Can extend stress test if ahead of schedule
- Always leave 1 minute for closing

### Troubleshooting on the Fly

- Stay calm if something breaks
- Have simulation mode ready to activate
- Acknowledge issues honestly
- Pivot to educational content if needed

## Post-Demo

### Immediate (0-10 minutes)

- Thank attendees
- Answer quick questions
- Direct people to booth
- Share vLLM meetup location again

### Follow-up (Day of)

- Note any technical issues encountered
- Collect feedback from attendees
- Debrief with Saša and booth team
- Share photos/videos with team

### Metrics to Capture

- Peak participant count
- Total requests processed
- Most popular model variant
- Technical issues encountered
- Audience questions/feedback
- Booth traffic after demo

## Emergency Contacts

**Technical Issues:**
- OpenShift AI Support: [contact]
- Red Hat IT Support: [contact]

**Event Coordination:**
- Juliana Furlow: jsweek@redhat.com, 678.447.9390
- Saša: sasa@redhat.com

## Resources

- Presenter dashboard: https://your-app-url.com/presenter
- Audience interface: https://your-app-url.com
- GitHub repo: https://github.com/MarkellR-RedHat/pytorch-quantization-demo
- OpenShift AI docs: https://docs.redhat.com/openshift-ai

---

**Remember:** The goal is to show quantization trade-offs visually and demonstrate OpenShift AI's multi-model serving capabilities. Keep it fun, educational, and engaging!

Good luck! 🚀
