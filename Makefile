.PHONY: shell build test dev-clean dev-image

shell:
	@bash scripts/dev_container.sh shell

build:
	@bash scripts/dev_container.sh build

test:
	@bash scripts/dev_container.sh test

dev-clean:
	@bash scripts/dev_container.sh clean

dev-image:
	@bash scripts/dev_container.sh image
