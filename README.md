# VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models
### [Project Page](https://videorepa.github.io) | [Paper](https://arxiv.org/abs/2505.23656)

> VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models\
> Xiangdong Zhang, Jiaqi Liao, Shaofeng Zhang, Fanqing Meng, Xiangpeng Wan, Junchi Yan, Yu Cheng \
> NeurIPS 2025

🎉 Our VideoREPA has been accepted by NeurIPS 2025. 

The code will be open-sourced in a few weeks. If you find our work helpful, feel free to give us a star ⭐ on GitHub for latest update.

### Introduction

<div align="center">
  <div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/7e65716b-27cd-45e1-b4df-1f4c1c7c3d33" alt="test" width="35%" />
    <img src="https://github.com/user-attachments/assets/1952c95f-5453-42d9-84ec-80f49565a961" alt="test" width="35%" />
  </div>
</div>

<p align="center">
  Figure 1. Evaluation of physics understanding on the Physion benchmark. The chance performance if 50%.
</p>

Recent advancements in text-to-video (T2V) diffusion models have enabled high-fidelity and realistic video synthesis. However, current T2V models often struggle to generate physically plausible content due to their limited inherent ability to accurately understand physics. We found that while the representations within T2V models possess some capacity for physics understanding, they lag significantly behind those from recent video self-supervised learning methods. **To this end, we propose a novel framework called VideoREPA, which distills physics understanding capability from video understanding foundation models into T2V models by aligning token-level relations.** This closes the physics understanding gap and enables more physics-plausible generation. Specifically, we introduce the Token Relation Distillation (TRD) loss, leveraging spatio-temporal alignment to provide soft guidance suitable for finetuning powerful pre-trained T2V models—a critical departure from prior representation alignment (REPA) methods. To our knowledge, VideoREPA is the first REPA method designed for finetuning T2V models and specifically for injecting physical knowledge. Empirical evaluations show that VideoREPA substantially enhances the physics commonsense of baseline method, CogVideoX, achieving significant improvement on relevant benchmarks and demonstrating a strong capacity for generating videos consistent with intuitive physics.

### Overview

<div align="center">
  <div style="display: flex; justify-content: space-between;">
    <img src="https://github.com/user-attachments/assets/4a55f50c-cc02-4467-8b84-4f83ed37869e" alt="test" width="70%" />
  </div>
</div>

<p align="center">
  Figure 2. Overview of VideoREPA.
</p>

Our VideoREPA enhances physics in T2V models by distilling physics knowledge from pre-trained SSL video encoders. We apply Token Relation Distillation (TRD) loss to align pairwise token similarities between video SSL representations and intermediate features in diffusion transformer blocks. Within each representation, tokens form spatial relations with other tokens in the same latent frame and temporal relations with tokens in other latent frames.

### ✅ TODO List

All code will be available later. We also appreciate it if you could give a star ⭐ to this repository for latest update.

- [x] Update the introduction to VideoREPA and the visual results comparison.
- [ ] Releasing the training and inference code.
- [ ] Uploading the checkpoints of VideoREPA.

### Qualitative Results

#### Teaser

<table align="center" style="width: 100%;">
  <tr>
    <th align="center" style="width: 25%;">CogVideoX</th>
    <th align="center" style="width: 25%;">CogVideoX+REPA loss</th>
    <th align="center" style="width: 25%;">VideoREPA</th>
    <th align="center" style="width: 25%;">Prompt</th>
  </tr>
  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/b0f6b65d-3b0b-4665-88a9-8fc81a23c613" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/00276b57-e3ea-4f30-b0a7-6522f4dedd31" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/92b51720-f4d8-4867-8c1c-3fb88e2f5e67" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Leather glove catching a hard baseball.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/a199b5ab-0829-41de-ab72-2e17ac66f069" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/e5d61296-0b9b-4567-aa27-b986234ce870" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/5acfd53c-83f7-4e18-bb37-b5eec8dcf226" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Maple syrup drizzling from a bottle onto pancakes.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/d93283f9-9dff-41b0-8837-d93d06d06356" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2be513e6-8a7f-4199-bb1e-c411fcda14ac" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/25dca90d-d91b-4ffe-8fc9-3c340c816d95" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Glass shatters on the floor.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/3095d887-4cfb-4726-8152-56d6aa72de40" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2b96fcde-f371-400f-9459-ca223d237c73" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/89299f65-fb1e-4013-969f-bc1c7e715523" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      A child runs and catches a brightly colored frisbee...
    </td>
  </tr>

</table>

#### Qualitative Comparison

<table align="center" style="width: 100%;">
  <tr>
    <th align="center" style="width: 25%;">HunyuanVideo</th>
    <th align="center" style="width: 25%;">CogVideoX</th>
    <th align="center" style="width: 25%;">VideoREPA</th>
    <th align="center" style="width: 25%;">Prompt</th>
  </tr>
  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/fa151629-0f96-411f-8b87-724c60165ac3" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/3b4a364e-8e58-4885-8660-9d5aa6dcf98e" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/994c20fb-d50f-452b-a882-1f3590d2105c" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Honey diffusing into warm milk.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/e2d66ee0-7b97-4b78-95a3-ffa8c5cfdd5b" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/908499fc-bc10-4068-a252-7d37826adc7b" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/c5e57f6b-48ab-4446-a0d5-bd9839499332" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      A cylindrical metal container rolls down a grassy hill...
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/4491422e-9801-4466-b756-e6a945219f95" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/37e568c2-7f39-444a-987a-8673929dcd11" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/f158a543-f96d-4cd3-a72d-02ce1fde0cfe" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Pencil rolls around on a flat desk.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2d3a1e52-9bfa-473c-9397-da3ba191a913" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2c5b98b7-f2b8-42fa-af85-7086e97f4039" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/575e0564-14ce-4587-8c1f-ad1c006c4364" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      A crane gently lifts a pallet of bricks.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/49d2f76f-9117-481b-9335-8e7d4fa7d22b" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/ab261289-f357-4d9b-af46-648371c6acb2" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/a1b13652-ad6b-491d-9055-4a38a28b7379" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      Two people stretch a bungee cord...
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/c4a43f53-e04c-4958-a511-830b748d9af6" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/09b58115-f0c2-4a1f-bbfc-c98e1e3ebe85" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 25%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/a43c1acf-4047-4420-a574-f2ce4e1e4d0d" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>    
    <td align="center" style="width: 25%;">
      A player kicks a soccer ball...
    </td>
  </tr>

</table>

#### CogVideoX vs. VideoREPA (ours)

<table align="center" style="width: 100%;">
  <tr>
    <th align="center" style="width: 33%;">CogVideoX</th>
    <th align="center" style="width: 33%;">VideoREPA</th>
    <th align="center" style="width: 34%;">Prompt</th>
  </tr>
  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/7843434a-39bf-4767-a59a-fdaa193c945b" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/472e560c-27bc-45d2-a9ae-d53f9739bf3a" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A credit card swipes through the machine.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/2fd68c46-2f80-4fda-a943-baf86733d473" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/82f6e3de-c8a7-4142-9829-2158873b685f" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A dog playfully bats at a balloon...
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/31015091-3804-485f-a0ae-03c3c1d11794" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/f1a07691-3167-476b-b192-4280c85a8504" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A hot iron pressing over a crumpled shirt.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/4cf13627-686c-43c3-bc67-beeac05d8729" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/bc2a3041-f08f-462e-b449-60f0fdba8b47" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A man throwing a stone across a river.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/6cc274a9-5596-4c49-9121-144451527d60" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/af248a5f-4e48-494d-bdec-95cec7957338" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A message contained bottle sails across the open sea.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/8dc36f57-b037-4065-896f-ce8c9e1bcb9b" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/5864ccd0-1b9c-46bd-aa01-e365a947d527" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A person mops up a puddle of water on a concrete floor...
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/098381aa-2a3a-464c-84ca-1c9f7af067c7" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/170102ff-8bdc-4c82-9d95-4e09c2da4525" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A person... pouring the remaining beer into a waiting shot glass.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/ad2d7608-effd-4445-8aa6-8667cdd998f2" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/02cf8795-5e44-4bb1-9706-030c8753fe25" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A single scull rower uses one oar to propel a boat.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/6646308d-0f19-4807-8a9f-08715fbc0c09" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/e8b2b785-887d-4514-96da-8236b17253d0" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A spray bottle sprays cleaning solution onto a countertop.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/55e2b720-b999-453e-9a54-6c8b69adc7f2" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/74ee294d-5951-4c8f-bd8a-424cdaa27c7c" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      ...a globe is poked, causing it to spin on its axis
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/43e64b7a-5613-4e93-b4b1-afa1b03bb6fd" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/84abb957-5113-42b2-9462-75031ada5ed2" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      Parallel bars are shown from a side view with an athlete performing dips.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/febf1c6d-4148-4817-8e12-77b8c8f69f4a" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/baebf5c0-ae12-49bb-9114-25321ba88753" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      Perfume spraying from a perfume bottle.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/ae0ee825-b5c7-4171-84a6-cae3003ced8c" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/1fbcfef4-eb62-4bad-9992-a9f08d583991" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      A person uses a low heat setting on the hairdryer to gently dry their fine hair.  
    </td>
  </tr>
    
  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/b283c2b3-4182-44b7-a364-6559e4096594" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/19dee5eb-6b7d-4efc-8565-0496661c797d" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      Detergent flowing into a bucket of water.
    </td>
  </tr>

  <tr>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/eabe7c0f-4c56-4f56-a585-da545a27aa82" controls autoplay loop muted></video>
    </td>
    <td align="center" style="width: 33%;">
      <video width="25%" controls src="https://github.com/user-attachments/assets/1dbcdcef-2a6e-4eba-a268-ab61d50bd410" controls autoplay loop muted>Your browser does not support the video tag.</video>
    </td>
    <td align="center" style="width: 34%;">
      Soap bar sliding off the ceramic dish.
    </td>
  </tr>

</table>

#### More Generated Videos

<table border="0" style="width: 100%; text-align: center; margin-top: 1px;">
  <tr>
    <td><video src="https://github.com/user-attachments/assets/6ae1ef3f-5cf8-491b-87bb-5c53384ae74e" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/de55cc3e-64ed-4961-bde4-ae84e1f47a93" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/59a6a046-6b6d-4c1c-8f50-af12a943d9f3" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>    
    <td><video src="https://github.com/user-attachments/assets/a529ad95-b0d6-40f7-aa87-1c2e0c68a923" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/d09520cf-d305-48b3-a8ab-c2619f343a84" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/d86ee8a1-3448-4344-89d4-4f04f60bb3dd" width="100%" controls autoplay loop muted></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/c377ad89-6324-486d-86fb-6489dec1d6af" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/2d01754d-5b4d-4f13-9c75-298af07701dd" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/88f5a3f1-a626-445e-a310-1f76753a6a74" width="100%" controls autoplay loop muted></video></td>
  </tr>

  <tr>
    <td><video src="https://github.com/user-attachments/assets/6fed6cc7-3b64-4821-aec3-8727a26f8a44" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/240fa721-08b9-4168-8659-024472ea7155" width="100%" controls autoplay loop muted></video></td>
    <td><video src="https://github.com/user-attachments/assets/040e37f2-ee6c-47ee-8953-bd7814695226" width="100%" controls autoplay loop muted></video></td>
  </tr>

</table>

## Contact

If you have any questions related to the code or the paper, feel free to email Xiangdong (`zhangxiangdong@sjtu.edu.cn`).

## Citation

```
@article{zhang2025videorepa,
  title={VideoREPA: Learning Physics for Video Generation through Relational Alignment with Foundation Models},
  author={Zhang, Xiangdong and Liao, Jiaqi and Zhang, Shaofeng and Meng, Fanqing and Wan, Xiangpeng and Yan, Junchi and Cheng, Yu},
  journal={arXiv preprint arXiv:2505.23656},
  year={2025}
}
```
