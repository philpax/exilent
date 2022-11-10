# exilent

A Discord bot, written in Rust, that provides a frontend to [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

## Setup

- Set up your Web UI instance as per the instructions, adding `--api --listen` to the launch arguments.
- Install Rust 1.65 or above using `rustup`.
- [Create a Discord application](https://discord.com/developers/applications) and fill it out with your own details.
- Go to `Bot` and create a new Bot.
  - Hit `Reset Token`, and copy the token it gives you somewhere.
- Go to `OAuth2 > URL Generator`, select `bot`, then select `Send Messages` and `Use Slash Commands`.
  - Go to the URL it generates, and then invite it to a server of your choice.
- In the exilent folder, create an `.env` folder with the following contents:

```dotenv
DISCORD_TOKEN = TOKEN_FROM_BEFORE
SD_URL = URL_OF_YOUR_WEBUI_INSTANCE
# optional:
SD_USER = USERNAME_FOR_UI
SD_PASS = PASSWORD_FOR_UI
```

- Run `cargo run` to start Exilent.
