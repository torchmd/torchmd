# Check that given variables are set and all have non-empty values,
# die with an error otherwise.
#
# Params:
#   1. Variable name(s) to test.
#   2. (optional) Error message to print.
check_defined = \
    $(strip $(foreach 1,$1, \
        $(call __check_defined,$1,$(strip $(value 2)))))
__check_defined = \
    $(if $(value $1),, \
      $(error Undefined variable: $1$(if $2, ($2))))

.PHONY: release docs

release:
	@:$(call check_defined, version, The release version)
	git tag -a v$(version) -m "v$(version) release"
	git push --tags origin v$(version)

docs:
	@:$(call check_defined, version, The release version)
	git tag -fa d$(version) -m "$(version) docs"
	git push -f --tags origin d$(version)
