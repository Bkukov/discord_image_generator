# bot.py
import os
import random
import discord
from dotenv import load_dotenv
from discord.ext import commands
from diffusers import StableDiffusionPipeline
import torch


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
GUILD = os.getenv("DISCORD_GUILD")

intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


@bot.event
async def on_ready():
    print(f"{bot.user.name} has connected to Discord!")


@bot.command(name="uwu", help="Gives you an random uwu emoji")
async def uwu_print(ctx):
    uwu_list = [
        "(◡ w ◡)",
        "(。U⁄ ⁄ω⁄ ⁄ U。)",
        "(„ᵕᴗᵕ„)",
        "(◡ ω ◡)",
        "( ͡U ω ͡U )",
        " ( ｡ᵘ ᵕ ᵘ ｡)",
        "(ᵘﻌᵘ)",
        "(灬´ᴗ`灬)",
    ]

    response = random.choice(uwu_list)
    await ctx.send(response)


@bot.command(name="generate", help="Returns a generated image")
@commands.has_any_role("Uncle Bob", "Gatekeeper")
async def uwu_print(ctx, prompt: str):
    torch.cuda.empty_cache()

    image = pipe(prompt).images[0]
    image.save(f"result.jpg")

    await ctx.send(file=discord.File("result.jpg"))


# @bot.command(name="create-channel")
# @commands.has_role("Uncle Bob")
# async def create_channel(ctx, channel_name="real-python"):
#     guild = ctx.guild
#     existing_channel = discord.utils.get(guild.channels, name=channel_name)
#     if not existing_channel:
#         print(f"Creating a new channel: {channel_name}")
#         await guild.create_text_channel(channel_name)


bot.run(TOKEN)
