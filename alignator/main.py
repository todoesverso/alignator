import typer


def main(name: str):
    typer.echo(f"Hello {name}")


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
