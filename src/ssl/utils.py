"""
Utility functions for SSL pipeline.
"""

import torch


def load_modality_batch(
    batch_files,
    modality,
    loader,
    transform,
    device,
):

    outputs = []
    mask = []


    for files in batch_files:

        imgs = []


        for f in files:

            try:

                img = loader.load(
                    f,
                    modality=modality
                )


                img = transform(
                    img
                )


                imgs.append(
                    img
                )


            except Exception:

                continue



        if imgs:

            outputs.append(
                torch.stack(imgs).mean(0)
            )

            mask.append(True)


        else:

            outputs.append(
                torch.zeros(
                    3,
                    224,
                    224
                )
            )

            mask.append(False)



    return (

        torch.stack(outputs).to(device),

        torch.tensor(
            mask,
            device=device
        )

    )