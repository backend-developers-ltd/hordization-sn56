from ch import constants as ch_cst

if ch_cst.RUNNING_IN_CONTAINER:
    import ch.download

    ch.download.patch()

    if ch_cst.USE_INPUT_VOLUME:
        from ch.download import (
            InputVolumeHfApi as HfApi,
            input_volume_hf_hub_download as hf_hub_download,
            input_volume_snapshot_download as snapshot_download,
        )
    else:
        from ch.download import (
            LoggingHfApi as HfApi,
            logging_hf_hub_download as hf_hub_download,
            logging_snapshot_download as snapshot_download,
        )

else:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
    )
