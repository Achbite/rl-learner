.PHONY: shell build deps dev-clean dev-image dev-refresh

shell:
	@bash scripts/dev_container.sh shell

build:
	@bash scripts/dev_container.sh build

deps:
	@bash scripts/prepare_dev_artifacts.sh

dev-clean:
	@bash scripts/dev_container.sh clean

dev-image:
	@bash scripts/dev_container.sh image

dev-refresh:
	@bash scripts/dev_container.sh refresh
