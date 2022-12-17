# exilent

A Discord bot, written in Rust, that provides a frontend to [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui).

Many thanks to [aiyabot](https://github.com/Kilvoctu/aiyabot), which paved the way for Discord bots for A1111. I just needed to build something to test [my client library](https://github.com/philpax/stable-diffusion-a1111-webui-client/) :)

## Setup

- Set up your Web UI instance as per the instructions, adding `--api --listen` to the launch arguments.
- Install Rust 1.65 or above using `rustup`.
- [Create a Discord application](https://discord.com/developers/applications) and fill it out with your own details.
- Go to `Bot` and create a new Bot.
  - Hit `Reset Token`, and copy the token it gives you somewhere.
- Go to `OAuth2 > URL Generator`, select `bot`, then select `Send Messages` and `Use Slash Commands`.
  - Go to the URL it generates, and then invite it to a server of your choice.
- Run `cargo run` to start Exilent. This will auto-generate a configuration file, and then quit.
- Fill in the configuration file with the required details.
- You can then run Exilent to your heart's content.
