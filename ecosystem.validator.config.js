module.exports = {
  apps: [
    {
      name: "natix_validator",
      script: "start_validator.sh",
      interpreter: "bash",
    },
    {
      name: "natix_cache_updater",
      script: "start_cache_updater.sh",
      interpreter: "bash",
    }
  ]
}