.PHONY: shell build dev-clean dev-image

shell:
	@bash scripts/dev_container.sh shell

build:
	@bash scripts/dev_container.sh build

dev-clean:
	@bash scripts/dev_container.sh clean

dev-image:
	@bash scripts/dev_container.sh image
